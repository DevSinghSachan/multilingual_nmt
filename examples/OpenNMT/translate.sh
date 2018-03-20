# Scripts to translate from OpenNMT-py

# For ja->en task
python translate.py -model models_ja_en_acc_63.38_ppl_9.24_e29.pt -src data/dev.ja -output pred.txt \
-verbose -report_bleu -gpu 0 -beam_size 5 -tgt data/dev.en -alpha 1

# For en->vi task
python translate.py -model models_en_vi_acc_52.39_ppl_15.53_e18.pt -src data/iwslt_en_vi/tst2012.en -output pred.txt \
-verbose -report_bleu -gpu 0 -beam_size 1 -tgt data/iwslt_en_vi/tst2012.vi -alpha 1



