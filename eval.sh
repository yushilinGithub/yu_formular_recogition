
export MODEL=/home/public/yushilin/formular/harvard/ft_formular_harDict_192_384_irpe/checkpoint_best.pt
#export MODEL=/home/public/yushilin/formular/harvard/ft_formular_harDict/checkpoint116.pt
export RESULT_PATH=/home/public/yushilin/formular/results/test
export BSZ=32


CUDA_VISIBLE_DEVICES=0 $(which fairseq-generate)  \
        --data-type formular --user-dir ./ --task text_recognition --input-size 192-768 \
        --beam 10 --scoring bleu --gen-subset test --batch-size ${BSZ} \
        --path ${MODEL} --results-path ${RESULT_PATH} --preprocess FM \
        --dict-path-or-url dictionary/latex_vocab_dict.txt  --skip-invalid-size-inputs-valid-test 
