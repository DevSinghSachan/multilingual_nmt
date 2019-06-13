### Transformer

This repository implements the `Transformer` model that was introduced in the paper *[Attention is All you Need](https://arxiv.org/abs/1706.03762)* as described in their
NIPS 2017 version: https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf

This codebase was also used for the bilingual and multilingual translation experiments for the paper "[Parameter Sharing Methods for Multilingual Self-Attentional Translation Models](https://arxiv.org/abs/1809.00252)"

The code in this repository implements the following features:
* positional encoding
* multi-head dot-product attention
* label smoothing
* warm-up steps based training of Adam optimizer
* shared weights of the embedding and softmax layers
* beam search with length normalization
* exponential moving average checkpoint of parameters

## Requirements
One can install the required packages from the requirements file.
```bash
pip install -r requirements.txt
```

## Usage
Please refer to scripts under "tools" directory for usage examples.

More details will be added soon.

## Dataset

1. Download the TED talks dataset as:
```bash
bash download_teddata.sh
``` 
This command will download, decompress, and will save the train, dev, and test splits of the TED talks under `data` directory.

2. One can use the script `ted_reader.py` to specify language pairs for both bilingual/multilingual translation tasks.
- For bilingual/multilingual translation, just specify the source and target languages as
```python
python ted_reader.py -s ja en zh fr ro -t en zh fr ro ja
``` 
- For multilingual translation, by default the training data will consist of the cartesian product of all the source and target language pairs. 
- If all possible combinations of the language pairs are not needed, then just use the option of `-ncp` 
```python
python ted_reader.py -s ja en zh fr ro -t en zh fr ro ja -ncp
```
- Above command will only create training data for the corresponding language pairs, i.e. [(ja, en), (en, zh), (zh, fr), (fr, ro), (fr, ja)]


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
