## Attention is All you Need (Transformer)

This repository implements the `transformer` model in *pytorch* framework which was introduced in the paper *[Attention is All you Need](https://arxiv.org/abs/1706.03762)* as described in their
NIPS 2017 version: https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf


The overall model architecture is as shown in the figure:

![][transformer]

[transformer]: img/transformer.png "Transformer Model"


The code in this repository implements the following features:
* Positional Encoding
* Multi-Head Dot-Product Attention
* Label Smoothing
* Warm-up steps based training of Adam Optimizer
* Shared weights of embedding and softmax layers
* Beam Search with length normalisation

## Software Requirements
* Python 3.6
* Pytorch v0.3.1 
* torchtext
* chainer
* numpy

One can install the above packages using the requirements file.
```bash
pip install -r requirements.txt
```


## Usage

### Step 1: Preprocessing:
Please refer to scripts under "tools" directory for usage examples.


## Dataset

Dataset Statistics included in `data` directory are:

| Dataset |Train|Dev|Test|
| --------------------------- |:-------:|------:|-------:|
| English-Vietnamese (IWSLT 2015) | 133,317 | 1,553 | 1,268  |
| English-German (TED talks)| 167,888 | 4,148 | 4,491 |
| English-Romanian (TED talks)| 180,484 | 3,904 | 4,631 |
| English-Dutch (TED talks)| 183,767 | 4,459 | 5,006 |

## Experiments
Bilingual Translation Tasks

| Dataset |This Repo |tensor2tensor| GNMT |
| --------------------------- |:-------:|:------:|:-------:|
| En -> Vi | 28.84 | 28.12 | 26.50 |
| En -> De | 29.31 | 28.68 | 27.01 |
| En -> Ro | 26.81 | 26.38 | 23.92 |
| En -> Nl | 32.42 | 31.74 | 30.64 |
| De -> En | 37.33 | 36.96 | 35.46 |
| Ro -> En | 37.00 | 35.45 | 34.77 |
| Nl -> En | 38.59 | 37.71 | 35.81 |


## Acknowledgements
* Thanks to the suggestions from Graham Neubig [@gneubig](https://github.com/neubig) and Matt Sperber [@msperber](https://github.com/msperber)
* The code in this repository was originally based and has been adapted from the [Sosuke Kobayashi](https://github.com/soskek)'s implementation in Chainer "https://github.com/soskek/attention_is_all_you_need".
* Some parts of the code were borrowed from [XNMT](https://github.com/neulab/xnmt/tree/master/xnmt) (based on [Dynet](https://github.com/clab/dynet)) and [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) (based on [Pytorch](https://github.com/pytorch/pytorch)).


If you find the code helpful, please cite our paper as:
```
@InProceedings{devendra2017multilingual,
  author = 	"Sachan, Devendra
		and Neubig, Graham,
  title = 	"Parameter Sharing Methods for Multilingual Self-Attentional Translation Models",
  booktitle = 	"Proceedings of the Third Conference on Machine Translation",
  year = 	"2018",
  publisher = 	"Association for Computational Linguistics",
  location = 	"Brussels, Belgium"
}
```