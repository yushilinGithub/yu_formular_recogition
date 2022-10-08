export MODEL=/path/of/the/trained_model
export RESULT_PATH=/path/of/result
#export DATA=/home/public/yushilin/formular/harvard/data/
export DATA=.
export BSZ=48

#swin_tiny_patch4_window7
CUDA_VISIBLE_DEVICES=0 $(which fairseq-generate)  \
        --data-type formular  --user-dir ./ --task text_recognition --input-size 112-336 \
        --beam 5 --scoring bleu --gen-subset valid --batch-size ${BSZ} \
        --path ${MODEL} --results-path ${RESULT_PATH} --preprocess FM \
        --dict dictionary/dictionary.txt  --skip-invalid-size-inputs-valid-test  \
        --data ${DATA}
