export MODEL=/home/public/yushilin/formular/result/okay/ft_formular_1m2latex_100k_longest_512/checkpoint_best.pt
export RESULT_PATH=.
#export DATA=/home/public/yushilin/formular/harvard/data/
export DATA=/home/public/yushilin/formular/im2latex-100k
export BSZ=48

#swin_tiny_patch4_window7
CUDA_VISIBLE_DEVICES=0 $(which fairseq-generate)  \
        --data-type formular  --user-dir ./ --task text_recognition --input-size 96-384 \
        --beam 5 --scoring bleu --gen-subset valid --batch-size ${BSZ} \
        --path ${MODEL} --results-path ${RESULT_PATH} --preprocess FM \
        --dict dictionary/im2latex_100k.txt --skip-invalid-size-inputs-valid-test  \
        --data ${DATA}
