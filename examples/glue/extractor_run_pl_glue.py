import argparse
import glob
import logging
import os
import time
import sys

sys.path.insert(0, '../../')


import sklearn.metrics
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss


from transformer_base import BaseTransformer, add_generic_args, generic_train
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes
from transformers import glue_processors as processors
from transformers import glue_tasks_num_labels
from transformer_anatomy.extractor import AutoExtractor
from transformer_anatomy.utils import find_top_n_layer
# from transformer_anatomy.tasks import dict_task_mapper

logger = logging.getLogger(__name__)
dict_task_mapper = {'sst-2':'SST2', 'sts-b':'STSBenchmark', 'mrpc':'MRPC'}

class Classifier(nn.Module):

    def __init__(self, encoder):
        super(Classifier, self).__init__()

        self.config = encoder.model.config

        if encoder.location == 'layer':
            self.pooled_output_size = len(encoder.pooling_position) * self.config.hidden_size
        elif encoder.location == 'head':
            self.pooled_output_size = len(encoder.pooling_position) * int(self.config.hidden_size/self.config.num_attention_heads)
        
        self.dense = nn.Linear(self.pooled_output_size, self.pooled_output_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.fc = nn.Linear(self.pooled_output_size, self.config.num_labels)

    def forward(
        self,
        pooled_output,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        pooled_output = self.activation(self.dense(pooled_output))
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)

        if labels is not None:
            if self.config.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            outputs = (loss, logits)

        return outputs  # (loss), logits, (hidden_states), (attentions)


class ExtractedGLUETransformer(BaseTransformer):

    mode = "base"

    def __init__(self, hparams):
        hparams.glue_output_mode = glue_output_modes[hparams.task]
        num_labels = glue_tasks_num_labels[hparams.task]

        # Create initial model
        super().__init__(hparams, num_labels, self.mode)

        if hparams.location is None:
            raise ValueError("location should be determined between 'head' and 'layer'. ")
        print(type(self.model))
        self.model = AutoExtractor.from_model(self.model, location=hparams.location, pooling_position=hparams.pooling_position)
        self.classifier = Classifier(self.model) 

    def forward(self, **inputs):
        labels = inputs.pop('labels')
        pooled_output = self.model(**inputs)
        #print('line96', pooled_output.shape)
        outputs = self.classifier(pooled_output, labels=labels, **inputs)
        return outputs

    def training_step(self, batch, batch_idx):
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}

        if self.config.model_type != "distilbert":
            inputs["token_type_ids"] = batch[2] if self.config.model_type in ["bert", "xlnet", "albert", "electra"] else None

        outputs = self(**inputs)
        loss = outputs[0]

        tensorboard_logs = {"loss": loss, "rate": self.lr_scheduler.get_last_lr()[-1]}
        return {"loss": loss, "log": tensorboard_logs}

    def prepare_data(self):
        "Called to initialize data. Use the call to construct features"
        args = self.hparams
        processor = processors[args.task]()
        self.labels = processor.get_labels()

        for mode in ["train", "dev"]:
            cached_features_file = self._feature_file(mode)
            if not os.path.exists(cached_features_file) and not args.overwrite_cache:
                logger.info("Creating features from dataset file at %s", args.data_dir)
                examples = (
                    processor.get_dev_examples(args.data_dir)
                    if mode == "dev"
                    else processor.get_train_examples(args.data_dir)
                )
                features = convert_examples_to_features(
                    examples,
                    self.tokenizer,
                    max_length=args.max_seq_length,
                    label_list=self.labels,
                    output_mode=args.glue_output_mode,
                )
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(features, cached_features_file)

    def load_dataset(self, mode, batch_size, n_worker=1):
        "Load datasets. Called after prepare data."

        # We test on dev set to compare to benchmarks without having to submit to GLUE server
        mode = "dev" if mode == "test" else mode

        cached_features_file = self._feature_file(mode)
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        if self.hparams.glue_output_mode == "classification":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        elif self.hparams.glue_output_mode == "regression":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

        return DataLoader(
            TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels),
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_worker
        )

    def validation_step(self, batch, batch_idx):
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}

        if self.config.model_type != "distilbert":
            inputs["token_type_ids"] = batch[2] if self.config.model_type in ["bert", "xlnet", "albert"] else None

        outputs = self(**inputs)
        tmp_eval_loss, logits = outputs[:2]
        preds = logits.detach().cpu().numpy()
        out_label_ids = inputs["labels"].detach().cpu().numpy()

        return {"val_loss": tmp_eval_loss.detach().cpu(), "pred": preds, "target": out_label_ids}

    def _eval_end(self, outputs):
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean().detach().cpu().item()
        preds = np.concatenate([x["pred"] for x in outputs], axis=0)

        if self.hparams.glue_output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif self.hparams.glue_output_mode == "regression":
            preds = np.squeeze(preds)

        out_label_ids = np.concatenate([x["target"] for x in outputs], axis=0)
        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        results = {**{"val_loss": val_loss_mean}, **compute_metrics(self.hparams.task, preds, out_label_ids)}

        ret = {k: v for k, v in results.items()}
        ret["log"] = results
        return ret, preds_list, out_label_list

    def validation_end(self, outputs: list) -> dict:
        ret, preds, targets = self._eval_end(outputs)
        logs = ret["log"]
        return {"val_loss": logs["val_loss"], "log": logs, "progress_bar": logs}

    def test_epoch_end(self, outputs):
        # updating to test_epoch_end instead of deprecated test_end
        ret, predictions, targets = self._eval_end(outputs)

        # Converting to the dic required by pl
        # https://github.com/PyTorchLightning/pytorch-lightning/blob/master/\
        # pytorch_lightning/trainer/logging.py#L139
        logs = ret["log"]
        # `val_loss` is the key returned by `self._eval_end()` but actually refers to `test_loss`
        return {"avg_test_loss": logs["val_loss"], "log": logs, "progress_bar": logs}

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        # Add NER specific options
        BaseTransformer.add_model_specific_args(parser, root_dir)
        parser.add_argument(
            "--max_seq_length",
            default=128,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )

        parser.add_argument(
            "--task", default="", type=str, required=True, help="The GLUE task to run",
        )

        parser.add_argument(
            "--data_dir",
            default=None,
            type=str,
            required=True,
            help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
        )

        parser.add_argument(
            "--evaluation_dir",
            default=None,
            type=str,
            required=True,
            help="The evaluation result data dir. Should contain head-wise or layer-wise evaluation results",
        )

        parser.add_argument(
            "--num_pooling",
            default=None,
            type=int,
            required=True,
            help="Number of pooled head or layer",
        )

        parser.add_argument(
            "--location",
            default=None,
            type=str,
            required=True,
            help="pooling output from 'head' or 'layer'",
        )

        parser.add_argument(
            "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
        )

        parser.add_argument(
            "--tags", nargs='+', type=str, help="experiment tags for neptune.ai", default=['FT', 'best-layer']
        )


        return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_generic_args(parser, os.getcwd())
    parser = ExtractedGLUETransformer.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()

    # If output_dir not provided, a folder will be generated in pwd
    if args.output_dir is None:
        args.output_dir = os.path.join("./results", f"{args.task}_{time.strftime('%Y%m%d_%H%M%S')}",)
        os.makedirs(args.output_dir)

    # Look-up evaluation results. 
    model_name = args.model_name_or_path
    model_name = model_name.split('/')[-1] if 'electra' in model_name else model_name
    if args.location == 'layer':
        args.pooling_position = find_top_n_layer(
            model_name=model_name,
            task=dict_task_mapper[args.task],
            dir_path=args.evaluation_dir,
            n_layer=args.num_pooling
        )
        print("================")
        print(args.pooling_position)
    elif args.location == 'head':
        pass
    else:
        raise ValueError("location should be 'layer' or 'head'. ")

    model = ExtractedGLUETransformer(args)
    trainer = generic_train(model, args)

    # Optionally, predict on dev set and write to output_dir
    if args.do_predict:
        checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "checkpointepoch=*.ckpt"), recursive=True)))
        model = model.load_from_checkpoint(checkpoints[-1])
        trainer.test(model)