### Attention is All you Need (Transformer)

This repository implements the `transformer` model that was introduced in the paper *[Attention is All you Need](https://arxiv.org/abs/1706.03762)* as described in their
NIPS 2017 version: https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf

This codebase was also used for the bilingual and multilingual translation experiments for the paper "[Parameter Sharing Methods for Multilingual Self-Attentional Translation Models](https://arxiv.org/abs/1809.00252)"

A high-level view of transformer model is as shown:


![transformer](https://raw.github.com/DevSinghSachan/multilingual_nmt/master/vis/transformer.png)


The code in this repository implements the following features:
* positional encoding
* multi-head dot-product attention
* label smoothing
* warm-up steps based training of Adam optimizer
* shared weights of the embedding and softmax layers
* beam search with length normalization
* exponential moving average checkpoint of parameters

## Software Requirements
* python 3.6
* pytorch v0.3.1 
* torchtext
* chainer
* numpy

One can install the above packages using the requirements file.
```bash
pip install -r requirements.txt
```

## Usage
Please refer to scripts under "tools" directory for usage examples.

More details will be added soon.

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

## Citation
If you find this code useful, please consider citing our paper as:
```
@InProceedings{devendra2018multilingual,
  author = 	"Sachan, Devendra
		and Neubig, Graham,
  title = 	"Parameter Sharing Methods for Multilingual Self-Attentional Translation Models",
  booktitle = 	"Proceedings of the Third Conference on Machine Translation",
  year = 	"2018",
  publisher = 	"Association for Computational Linguistics",
  location = 	"Brussels, Belgium"
}
```

## Acknowledgements
* Thanks to the suggestions from Graham Neubig [@gneubig](https://github.com/neubig) and Matt Sperber [@msperber](https://github.com/msperber)
* The code in this repository was originally based and has been adapted from the [Sosuke Kobayashi](https://github.com/soskek)'s implementation in Chainer "https://github.com/soskek/attention_is_all_you_need".
* Some parts of the code were borrowed from [XNMT](https://github.com/neulab/xnmt/tree/master/xnmt) (based on [Dynet](https://github.com/clab/dynet)) and [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) (based on [Pytorch](https://github.com/pytorch/pytorch)).
