## Multilingual Translation

This codebase was used for the multilingual translation experiments for the paper "[Parameter Sharing Methods for Multilingual Self-Attentional Translation Models](https://arxiv.org/abs/1809.00252), WMT-EMNLP 2018".

The multilingual model is based on the *[Transformer](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)* model and also contains the following features:
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

## Dataset

* Download the TED talks dataset as:
```bash
bash download_teddata.sh
``` 
This command will download, decompress, and will save the train, dev, and test splits of the TED talks under `data` directory.

* One can use the script `ted_reader.py` to specify language pairs for both bilingual/multilingual translation tasks.
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

* For evaluating the multiingual model, one can generate the test set for each bilingual pair using the above command.  

## Instructions
For convenience, there are some example shell scripts under tools directory
* Bilingual Translation (NS)
```bash
bash tools/bpe_pipeline_bilingual.sh src_lang tgt_lang
```

- Fully Shared Multilingual Translation (FS)
```bash
bash tools/bpe_pipeline_fully_shared_multilingual.sh src_lang tgt_lang1 tgt_lang2 
```

- Partial Sharing Multilingual Translation (PS)
```bash
bash tools/bpe_pipeline_MT.sh src_lang tgt_lang1 tgt_lang2 share_sublayer share_attn
```
An example of sharing the Key(k), Query(q) in both the attention layers (Self, Source) 
```bash
bash tools/bpe_pipeline_MT.sh src_lang tgt_lang1 tgt_lang2 k,q self,source
```

## Experiments

* **Dataset Statistics**

| Dataset | Train | Dev | Test |
| --------------------------- |:-------:|------:|-------:|
| English-Vietnamese (IWSLT 2015) | 133,317 | 1,553 | 1,268  |
| English-German (TED talks)| 167,888 | 4,148 | 4,491 |
| English-Romanian (TED talks)| 180,484 | 3,904 | 4,631 |
| English-Dutch (TED talks)| 183,767 | 4,459 | 5,006 |


* **Bilingual Translation Tasks**

| language pairs |this repo |tensor2tensor| GNMT |
| --------------------------- |:-------:|:------:|:-------:|
| En -> Vi (IWSLT 2015) | 28.84 | 28.12 | 26.50 |
| En -> De | 29.31 | 28.68 | 27.01 |
| En -> Ro | 26.81 | 26.38 | 23.92 |
| En -> Nl | 32.42 | 31.74 | 30.64 |
| De -> En | 37.33 | 36.96 | 35.46 |
| Ro -> En | 37.00 | 35.45 | 34.77 |
| Nl -> En | 38.59 | 37.71 | 35.81 |


* **Multilingual Translation Tasks**

| Method | En->De+Tr | En->De+Ja | En->Ro+Fr | En->De+Nl |
| :---- |:-------:|:------:|:-------:|:----:|
|        | ->De  ->Tr | ->De ->Ja | ->Ro ->Fr | ->De ->Nl |
| GNMT NS        |27.01 16.07|27.01 16.62|24.38 40.50|27.01 30.64|
| GNMT FS        |29.07 18.09|28.24 17.33|26.41 42.46|28.52 31.72|
| Transformer NS |29.31 18.62|29.31 17.92|26.81 42.95|29.31 32.43|
| Transformer FS |28.74 18.69|29.68 18.50|**28.52 44.28**|30.45 33.69|
| Transformer PS |**30.71 19.67**|**30.48 19.00**|27.58 43.84|**30.70 34.05**|


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