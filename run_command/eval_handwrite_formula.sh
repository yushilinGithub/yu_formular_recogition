MODEL=/modelpath/
RESULT_PATH=/save the trained model/

#export DATA=/home/public/yushilin/formular/harvard/data/
DATA=.
BSZ=32

#swin_tiny_patch4_window7
CUDA_VISIBLE_DEVICES=1 $(which fairseq-generate)  \
        --data-type formularHandWrite  --user-dir ./ --task text_recognition --input-size 96-384 \
        --beam 10 --scoring acc_ed --gen-subset valid --batch-size ${BSZ} \
        --path ${MODEL} --results-path ${RESULT_PATH} --preprocess Syn \
        --dict dictionary/handWriteFormulaComp.txt  --skip-invalid-size-inputs-valid-test  \
        --data ${DATA}
