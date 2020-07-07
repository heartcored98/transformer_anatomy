## [Transformer Anatomy]: Roles and Utilization of Attention Heads in Transformer-based Neural Language Models (ACL 2020)  

Official Pytorch implementation of **Transformer Anatomy** | [Paper](https://www.aclweb.org/anthology/2020.acl-main.311/)  

Jae-young Jo<sup>1,2</sup>, Sung-hyon Myaeng<sup>1</sup>  

<sup>1</sup> <sub>KAIST</sub>  
<sup>2</sup> <sub>Dingbro, Inc</sub>  


Sentence encoders based on the transformer architecture have shown promising results on various natural language tasks. The main impetus lies in the pre-trained neural language models that capture long-range dependencies among words, owing to multi-head attention that is unique in the architecture. However, little is known for how linguistic properties are processed, represented, and utilized for downstream tasks among hundreds of attention heads inside the pre-trained transformer-based model. For the initial goal of examining the roles of attention heads in handling a set of linguistic features, we conducted a set of experiments with ten probing tasks and three downstream tasks on four pre-trained transformer families (GPT, GPT2, BERT, and ELECTRA). Meaningful insights are shown through the lens of heat map visualization and utilized to propose a relatively simple sentence representation method that takes advantage of most influential attention heads, resulting in additional performance improvements on the downstream tasks.


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


## TODO 
[X] Clean up directory structure  
[X] Recover layer-wise, head-wise | probing-task, downstream-task 
[X] Recover fine-tuning and solve dependency with legacy code  
[ ] Update embedding caching algorithm -> pickle to parquet  
[ ] Run multiple trial based on ray (sharing cache object)
[ ] Refactoring Anatomy Wrapper class  
[X] Add experiments result access class   
[ ] Semantic / Syntactic Visualization  
