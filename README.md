# PGT: Pseudo Relevance Feedback Using a Graph-based Transformer

Repo of code and data for ECIR 2021 short paper "PGT: Pseudo Relevance Feedback Using a Graph-based Transformer"

Abstract: Most research on pseudo relevance feedback (PRF) has been done in vector space and probabilistic retrieval models. This paper shows that Transformer-based rerankers can also benefit from the extra context that PRF provides. It presents PGT, a graph-based Transformer that sparsifies attention between graph nodes to enable PRF while avoid- ing the high computational complexity of most Transformer architec- tures. Experiments show that PGT improves upon non-PRF Transformer reranker, and it is at least as accurate as Transformer PRF models that use full attention, but with lower computational costs.

## Environment


## Data 
Please download `data.zip` from our virtual appendix [here](http://boston.lti.cs.cmu.edu/appendices/ECIR20-HongChien-Yu/downloads). Unzip the file, and place the `data` folder as it is in this repo. 

### Folder Structure 
The `data.zip` folder is structured as follows, where ${i} is the file index. The training set is too large to fit into the RAM, so we break it into 26 blocks and read 1 block at a time to save the working memory. 
```
data 
  - trec 
    - manual-qrels-pass.txt                       # gold standard query relevance judgements for TREC 2019 dev set 
  - top7                                          # the raw data and the tokenized data where 7 feedback documents are used (k=7)
    - bm25_train                                  # training data obtained using BM25 as the initial ranker 
      - train.graph.top7.json${i}                 # raw training data 
    - bm25_test                                   # test data obtained using BM25 as the initial ranker 
      - trec.top7.test.graph.json                 # raw test data 
      - pids.tsv                                  # qid \t pid (each qid corresponds to 1000 top initial retrieval pids) 
    - crm_test                                    # same as bm25_test
      - ... 
```

### Data Format 
`train.graph.top7.json${i}$` formats the graph inputs as json structures. Each line in the file is one graph corresponding to one candidate document of one query. For example 
```
{"qid": "597347",                                 # query id 
"query": "what color are the four circles ...",     
"candidate": ["bee", "##t", "varieties", ...],    # byte-pair-encoded candidate document  
"label": 0,                                       # binary relevance label of the candidate document 
"node": [
  {"node_id": 0,                                          
   "passage": ["each", "row", "should", ...],     # byte-pair-encoded feedback document 0 
   "label": 0},                                   # binary relevance label of the feedback document (not used in our experiments)
   {"node_id": 1, ...}
   {"node_id": 2, ...}, 
    ...
   {"node_id": 6, ...}
 ]
}
```

## Data Preprocessing 
Tokenizing the data takes time. The following script tokenizes the training data and saves them into .cache files, which can be read by the software once detected. You may run the script on CPU (requires the CPU-version `dgl`) and in parallel across `i`s. 
```
for i in {0..25}
do
  python ./main.py \
    --cf=config.json \
    --load_train \
    --data_path=./data/top7/bm25_train/train.graph.top7.json${i} 
done 
```
Similarly, to tokenize test data, run: 

```
python ./main.py \
  --cf=config.json \
  --load_test \
  --data_path=./data/top7/bm25_test/trec.top7.test.graph.json
```
  
## Training 
The following script trains the model in a distributed manner on 2 GPUs. 
```
python -m torch.distributed.launch --nproc_per_node=2 ./main.py \
  --checkpoint=25000 \
  --train \
  --cf=config.json \
  --distributed 
```
Single-GPU training is also possible with 
```
python ./main.py \
  --checkpoint=25000 \
  --train \
  --cf=config.json \
```

## Testing 
The following script tests the model on TREC dev set. Change `config["system"]["test_data"]` to use either BM25 or CRM as the initial ranker. 
```
python ./main.py \
  --cf=config.json \
  --test \
  --model_path=./top7/epoch1_final.pt \
  --test_output=./top7/epoch1_final.out
```
Then run 
```
python ./pred_to_ranking.py 
  --prefix=./top7/epoch1_final  \
  --initial_ranker=bm25  \           
  --trec_eval_path=your_trec_eval_path/trec_eval  \
  --gold_path=./data/trec/manual-qrels-pass.txt 
```
which produces `./top7/epoch1_final.csv` reporting MAP and NDCG scores at different reranking depths.

## Configurations 
`config.json` sets the hyperparameter and dataset paths. 
`config["ablation"]` controls the graph structure as described in Section 4.3 of our paper. You can use one of the following five strings: `base` (PGT base), `exp1` (base w/o pre d_c), `exp2` (base w/o pre q, d_c), `exp3` (base w/o node d_c), and `exp4` (base w/o node q, d_c). 

## Trained Models 
We provide models trained using 7 feedback documents (k=7) with different graph structures [here](http://boston.lti.cs.cmu.edu/appendices/ECIR20-HongChien-Yu/downloads/models). 
