#!/usr/bin/env bash

STAT_MT_URL="http://data.statmt.org/wmt18/translation-task/"
NC_TRAIN_DATASETS=${STAT_MT_URL}"training-parallel-nc-v13.tgz"
NC_DEV_DATASETS=${STAT_MT_URL}"dev.tgz"
NC_TEST_DATASETS=${STAT_MT_URL}"test.tgz"

UN_TRAIN_DATASETS="https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/UNv1.0.en-zh.tar.gz"
CWMT_TRAIN_DATASETS="https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/cwmt.tgz"

wget ${NC_TRAIN_DATASETS}
wget ${NC_DEV_DATASETS}
wget ${NC_TEST_DATASETS}
wget ${UN_TRAIN_DATASETS}
wget ${CWMT_TRAIN_DATASETS}

rm train.en train.zh

s1="training-parallel-nc-v13/news-commentary-v13.zh-en.en"
t1="training-parallel-nc-v13/news-commentary-v13.zh-en.zh"
cat $s1 >> train.en
cat $t1 >> train.zh

s2="UNv1.0en-zh/UNv1.0.en-zh.en"
t2="UNv1.0en-zh/UNv1.0.en-zh.zh"
cat $s2 >> train.en
cat $t2 >> train.zh

s3="cwmt/casia2015/casia2015_en.txt"
t3="cwmt/casia2015/casia2015_ch.txt"
cat $s3 >> train.en
cat $t3 >> train.zh

s4="cwmt/casict2015/casict2015_en.txt"
t4="cwmt/casict2015/casict2015_ch.txt"
cat $s4 >> train.en
cat $t4 >> train.zh

s5="cwmt/neu2017/NEU_en.txt"
t5="cwmt/neu2017/NEU_cn.txt"
cat $s5 >> train.en
cat $t5 >> train.zh

s6="cwmt/datum2015/datum_en.txt"
t6="cwmt/datum2015/datum_ch.txt"
cat $s6 >> train.en
cat $t6 >> train.zh

s7="cwmt/datum2017/Book1_en.txt"
t7="cwmt/datum2017/Book1_cn.txt"
cat $s7 >> train.en
cat $t7 >> train.zh

s8="cwmt/datum2017/Book2_en.txt"
t8="cwmt/datum2017/Book2_cn.txt"
cat $s8 >> train.en
cat $t8 >> train.zh

s9="cwmt/datum2017/Book3_en.txt"
t9="cwmt/datum2017/Book3_cn.txt"
cat $s9 >> train.en
cat $t9 >> train.zh

s10="cwmt/datum2017/Book4_en.txt"
t10="cwmt/datum2017/Book4_cn.txt"
cat $s10 >> train.en
cat $t10 >> train.zh

s11="cwmt/datum2017/Book5_en.txt"
t11="cwmt/datum2017/Book5_cn.txt"
cat $s11 >> train.en
cat $t11 >> train.zh

s12="cwmt/datum2017/Book6_en.txt"
t12="cwmt/datum2017/Book6_cn.txt"
cat $s12 >> train.en
cat $t12 >> train.zh

s13="cwmt/datum2017/Book7_en.txt"
t13="cwmt/datum2017/Book7_cn.txt"
cat $s13 >> train.en
cat $t13 >> train.zh

s14="cwmt/datum2017/Book8_en.txt"
t14="cwmt/datum2017/Book8_cn.txt"
cat $s14 >> train.en
cat $t14 >> train.zh

s15="cwmt/datum2017/Book9_en.txt"
t15="cwmt/datum2017/Book9_cn.txt"
cat $s15 >> train.en
cat $t15 >> train.zh

s16="cwmt/datum2017/Book10_en.txt"
t16="cwmt/datum2017/Book10_cn.txt"
cat $s16 >> train.en
cat $t16 >> train.zh

s17="cwmt/datum2017/Book11_en.txt"
t17="cwmt/datum2017/Book11_cn.txt"
cat $s17 >> train.en
cat $t17 >> train.zh

s18="cwmt/datum2017/Book12_en.txt"
t18="cwmt/datum2017/Book12_cn.txt"
cat $s18 >> train.en
cat $t18 >> train.zh

s19="cwmt/datum2017/Book13_en.txt"
t19="cwmt/datum2017/Book13_cn.txt"
cat $s19 >> train.en
cat $t19 >> train.zh

s20="cwmt/datum2017/Book14_en.txt"
t20="cwmt/datum2017/Book14_cn.txt"
cat $s20 >> train.en
cat $t20 >> train.zh

s21="cwmt/datum2017/Book15_en.txt"
t21="cwmt/datum2017/Book15_cn.txt"
cat $s21 >> train.en
cat $t21 >> train.zh

s22="cwmt/datum2017/Book16_en.txt"
t22="cwmt/datum2017/Book16_cn.txt"
cat $s22 >> train.en
cat $t22 >> train.zh

s23="cwmt/datum2017/Book17_en.txt"
t23="cwmt/datum2017/Book17_cn.txt"
cat $s23 >> train.en
cat $t23 >> train.zh

s24="cwmt/datum2017/Book18_en.txt"
t24="cwmt/datum2017/Book18_cn.txt"
cat $s24 >> train.en
cat $t24 >> train.zh

s25="cwmt/datum2017/Book19_en.txt"
t25="cwmt/datum2017/Book19_cn.txt"
cat $s25 >> train.en
cat $t25 >> train.zh

s26="cwmt/datum2017/Book20_en.txt"
t26="cwmt/datum2017/Book20_cn.txt"
cat $s26 >> train.en
cat $t26 >> train.zh

# Convert SGM files
echo "Cloning moses for data processing"
git clone https://github.com/moses-smt/mosesdecoder.git

# Convert newsdev2017 data into raw text format
ds1="dev/newsdev2017-enzh-src.en.sgm"
dt1="dev/newsdev2017-enzh-ref.zh.sgm"
mosesdecoder/scripts/ems/support/input-from-sgm.perl < $ds1 > dev.en
mosesdecoder/scripts/ems/support/input-from-sgm.perl < $dt1 > dev.zh

# Convert newstest2018 data into raw text format
ts1="test/newstest2018-enzh-src.en.sgm"
tt1="test/newstest2018-enzh-ref.zh.sgm"
mosesdecoder/scripts/ems/support/input-from-sgm.perl < $ts1 > test.en
mosesdecoder/scripts/ems/support/input-from-sgm.perl < $tt1 > test.zh
