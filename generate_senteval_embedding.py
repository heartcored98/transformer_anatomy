import sys

PATH_BERT = '../pytorch-pretrained-BERT'
sys.path.insert(0, PATH_BERT)
from pytorch_pretrained_bert import BertTokenizer, BertModel


PATH_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data/'
PATH_TO_CACHE = './cache/'
sys.path.insert(0, PATH_SENTEVAL)
import senteval

from encoder import BERTEncoder, GPTEncoder
from encoder.single_head_exp import *


tasks = ['Length', 'WordContent', 'Depth', 'TopConstituents',
         'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
         'OddManOut', 'CoordinationInversion', 'CR', 'MR', 
         'MPQA', 'SUBJ', 'SST2', 'SST5', 
         'TREC', 'MRPC', 'SNLI', 'SICKEntailment', 
         'SICKRelatedness', 'STSBenchmark', 'ImageCaptionRetrieval', 'STS12',
         'STS13', 'STS14', 'STS15', 'STS16']


if __name__ == '__main__':

    # ====== Generate Embedding of Large Model ====== #
    parser = argparse.ArgumentParser(description='Evaluate BERT')
    parser.add_argument("--device", type=list, default=[1,2,3,4,5,6,7])
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--kfold", type=int, default=5)
    parser.add_argument("--usepytorch", type=bool, default=True)
    parser.add_argument("--task_path", type=str, default='./SentEval/data/')
    parser.add_argument("--cache_path", type=str, default='./cache/')
    parser.add_argument("--result_path", type=str, default='./encoder_test_results/')
    parser.add_argument("--optim", type=str, default='rmsprop')
    parser.add_argument("--cbatch_size", type=int, default=512)
    parser.add_argument("--tenacity", type=int, default=3)
    parser.add_argument("--epoch_size", type=int, default=2)
    parser.add_argument("--model_name", type=str, default='openai-gpt')

    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--layer", nargs='+', type=int, default=0)
    parser.add_argument("--head", nargs='+', type=int, default=0) #8, 15
    parser.add_argument("--location", type=str, default='head') #8, 15
    parser.add_argument("--head_size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--nhid", type=int, default=0)


    args = parser.parse_args()
    args.seed = 123
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.device)

    num_exp = 10
    print("======= Benchmark Configuration ======")
    print("Args: ", args)
    print("Device: ", args.device)
    print("model name: ", args.model_name)
    print("Task: ", tasks[args.task])
    print("location: ", args.location)
    print("Total Exps: ", num_exp)
    print("======================================")

    cnt = 0
    if args.model_name in ['bert-base-uncased', 'bert-large-uncased'] :
        model = BERTEncoder(model_name=args.model_name, encode_capacity=args.batch_size)
    elif args.model_name == 'openai-gpt':
        model = GPTEncoder(encode_capacity=args.batch_size)

    with tqdm(total=num_exp, file=sys.stdout) as pbar:
        for task in range(10):
            args.task = tasks[task]

            exp_result = experiment(model, args.task, deepcopy(args))
            print('task', exp_result['acc'], args.task)

            pbar.set_description('P: %d' % (1 + cnt))
            pbar.update(1)
            cnt += 1


