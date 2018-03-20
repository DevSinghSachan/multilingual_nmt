AIAYN="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

cd ..

# Preprocess
python ${AIAYN}/preprocess.py -i ${AIAYN}/data/ja_en -s-train train-big.ja -t-train train-big.en -s-valid dev.ja -t-valid dev.en -s-test test.ja -t-test test.en --save_data demo

# Train
python ${AIAYN}/train.py -i ${AIAYN}/data/ja_en --data demo --batchsize 60 --tied --beam_size 5 --dropout 0.1 --epoch 40 \
--layers 1 --multi_heads 8 --gpu 0 --model_file "${AIAYN}/results/model.ckpt"

# Predict
python ${AIAYN}/translate.py -i ${AIAYN}/data/ja_en --data demo --batchsize 60 --beam_size 5 \
--model_file "${AIAYN}/results/model.ckpt" --src ${AIAYN}/data/ja_en/test.ja
