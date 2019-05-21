## Transformer Anatomy: Roles of Attention Heads in Transformer-based Neural Language Models (ACL 2019 Submission)  

### Toolkit for finding and analyzing important attention heads in transformer-based models  

Majority of researchers would agree that the BERT and other transformer models outperform previous models (LSTM, ELMo, etc..). However, most of us still does not understand well about how the transformers digest given input text internally. Therefore, this repository is a toolkit includes

- Evaluate pre-trained(or just trained) transformer model with given downstream task in order to find out **superior attention heads** or encoder-layers including intuitive visualization within heatmap plot. 

- Construct **better sentence representation** without giving a change to pre-trained transformer by concatenating the output vectors from the superior attention heads.  

# 

### DEMO #1 - Inspecting Internal Linguistic Information Handling Inside Transformers  

![Image](https://github.com/heartcored98/Transformer_Anatomy/blob/master/imgs/showcase1.png?raw=true)

<p align="center"> 
  Heatmaps of attention head-wise evaluation on the five sentence probing tasks with pre-trained <b>BERT BASE</b> model.   
</p>  

  Each column correspond to following tasks (Length, Depth, BigramShift, CoordinationInversion, Tense from the left). For each heatmap, x-axis and y-axis show the index values of the attention heads and the layer numbers (the lower, the closer to the initial input), respectively. 
  
  The brighter the color, the higher the accuracy for the attention head and hence more important for the task. Note that the attention heads in the same layer are ordered by their classification accuracy values (i.e. an attention head with the highest accuracy on a layer is at the left-most location) This heatmap could give you intuitive understanding of where the task-related information is handled inside the sentence encoder. 
  
We can observe following internal tendency of **BERT BASE** along various linguistic features.  
- Surface and syntactic related information(Length and Depth) is usually captured from the attention heads close to the input layer.  
- Word or clause order related information(BigramShift and CoordinationInversion) is well captured from the attention heads located in the middle layer. 
- Semantic related information(Tense) is well captured from the attention heads close to the output layer.  

#  

### DEMO #2 - Boosting Downstream Task Performance  
##### (Extracting Essential Hidden Sentence Representation)    

![Image](https://github.com/heartcored98/Transformer_Anatomy/blob/master/imgs/showcase2_downstream_heatmap.png?raw=true)

<p align="center"> 
  Heatmaps of attention head-wise evaluation on the four downstream tasks with pre-trained <b>BERT BASE</b> model.   
</p>  

Each column correspond to following tasks (SST5, TREC, SICKEntailment, MRPC from the left). 

We can observe following internal tendency of **BERT BASE** along various downstream-task related information.  
- Surface and syntactic related information(Length and Depth) is usually captured from the attention heads close to the input layer.  
- Word or clause order related information(BigramShift and CoordinationInversion) is well captured from the attention heads located in the middle layer. 
- Semantic related information(Tense) is well captured from the attention heads close to the output layer.  
   According to our results, these reconstructed sentence representation outperform compare to the sentecne representation from the last layer.      

Futhermore, we show how the downstream performance could be increased by just pulling internal outperforming hidden representation and using it as sentence representation. 

## Reference  
These implementations is largely based on the following implementations. 
- [huggingface's pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT)  
- [facebookresearch's SentEval](https://github.com/facebookresearch/SentEval)  

**huggingface's pytorch-pretrained-BERT**  provides the pre-trained transformer encoders not only BERT but also GPT/GPT2 and Transformer-XL models. The results of the paper is produced by using these implementation with **slight change**.  

**facebookresearch's SentEval** provides toolkit for benchmarking the sentence embedding of given sentence encoder model on 17 downstream tasks and 10 probing tasks. The benchmarking result of the paper is produced by using these pre-implemented datasets.