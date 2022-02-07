
export MODEL=/home/public/yushilin/formular/harvard/ft_formular_harDict_192_384_swin/checkpoint_best.pt
#export MODEL=/home/public/yushilin/formular/harvard/ft_formular_harDict/checkpoint116.pt
export RESULT_PATH=/home/public/yushilin/formular/results/test_swin
export DATA=/home/public/yushilin/formular/harvard/data
export BSZ=32

#swin_tiny_patch4_window7
CUDA_VISIBLE_DEVICES=0 $(which fairseq-generate)  \
        --data-type formular  --user-dir ./ --task text_recognition --input-size 224-784 \
        --beam 5 --scoring bleu --gen-subset test --batch-size ${BSZ} \
        --path ${MODEL} --results-path ${RESULT_PATH} --preprocess FM \
        --dict-path-or-url dictionary/latex_vocab_dict.txt  --skip-invalid-size-inputs-valid-test  \
        --data ${DATA} 
