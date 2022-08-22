# Work from Lab Course Legal Data Analysis: Hierarchical Multi Label Text Classification. Xi Sun with supervisor Santosh Tokala.
# This code repository is based on the work from Incorporating Hierarchy into Text Encoder: a Contrastive Learning Approach for Hierarchical Text Classification


## Requirements

* Python >= 3.6
* torch >= 1.6.0
* transformers == 4.2.1
* fairseq >= 0.10.0
* torch-geometric == 1.7.2
* torch-scatter == 2.0.8
* torch-sparse ==  0.6.12

## Preprocess

Please download the original dataset and then use these scripts.

### WebOfScience

The original dataset can be acquired in [the repository of HDLTex](https://github.com/kk7nc/HDLTex). Preprocess code could refer to [the repository of HiAGM](https://github.com/Alibaba-NLP/HiAGM) and we provide a copy of preprocess code here.

```shell
cd ./data/WebOfScience
python preprocess_wos.py
python data_wos.py
```


### RCV1-V2

The preprocess code could refer to the [repository of reuters_loader](https://github.com/ductri/reuters_loader) and we provide a copy here. The original dataset can be acquired [here](https://trec.nist.gov/data/reuters/reuters.html) by signing an agreement.

```shell
cd ./data/rcv1
python preprocess_rcv1.py
python data_rcv1.py
```

## Train

```
usage: train.py [-h] [--lr LR] [--data {WebOfScience,nyt,rcv1}] [--batch BATCH] [--early-stop EARLY_STOP] [--device DEVICE] --name NAME [--update UPDATE] [--warmup WARMUP] [--contrast CONTRAST] [--graph GRAPH] [--layer LAYER]
                [--multi] [--lamb LAMB] [--thre THRE] [--tau TAU] [--seed SEED] [--wandb]

optional arguments:
  -h, --help            show this help message and exit
  --lr LR               Learning rate.
  --data {WebOfScience,nyt,rcv1}
                        Dataset.
  --batch BATCH         Batch size.
  --early-stop EARLY_STOP
                        Epoch before early stop.
  --device DEVICE		cuda or cpu. Default: cuda
  --name NAME           A name for different runs.
  --update UPDATE       Gradient accumulate steps
  --warmup WARMUP       Warmup steps.
  --contrast CONTRAST   Whether use contrastive model. Default: True
  --graph GRAPH         Whether use graph encoder. Default: True
  --layer LAYER         Layer of Graphormer.
  --multi               Whether the task is multi-label classification. Should keep default since all 
  						datasets are multi-label classifications. Default: True
  --lamb LAMB           lambda
  --thre THRE           Threshold for keeping tokens. Denote as gamma in the paper.
  --tau TAU             Temperature for contrastive model.
  --seed SEED           Random seed.
  --wandb               Use wandb for logging.
```

Checkpoints are in `./checkpoints/DATA-NAME`. Two checkpoints are kept based on macro-F1 and micro-F1 respectively 
(`checkpoint_best_macro.pt`, `checkpoint_best_micro.pt`).

e.g. Train on `WebOfScience` with `batch=12, lambda=0.05, gamma=0.02`. Checkpoints will be in `checkpoints/WebOfScience-test/`.

```shell
python train.py --name test --batch 12 --data WebOfScience --lambda 0.05 --thre 0.02
```


## Test

```
usage: test.py [-h] [--device DEVICE] [--batch BATCH] --name NAME [--extra {_macro,_micro}]

optional arguments:
  -h, --help            show this help message and exit
  --device DEVICE
  --batch BATCH         Batch size.
  --name NAME           Name of checkpoint. Commonly as DATA-NAME.
  --extra {_macro,_micro}
                        An extra string in the name of checkpoint. Default: _macro
```

Use `--extra _macro` or `--extra _micro`  to choose from using `checkpoint_best_macro.pt` or`checkpoint_best_micro.pt` respectively.

e.g. Test on previous example.

```shell
python test.py --name WebOfScience-test
```


## Citation

```
@inproceedings{wang-etal-2022-incorporating,
    title = "Incorporating Hierarchy into Text Encoder: a Contrastive Learning Approach for Hierarchical Text Classification",
    author = "Wang, Zihan  and
      Wang, Peiyi  and
      Huang, Lianzhe  and
      Sun, Xin  and
      Wang, Houfeng",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.491",
    pages = "7109--7119",
}
```
