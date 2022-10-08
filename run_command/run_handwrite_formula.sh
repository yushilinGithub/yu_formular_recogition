
export MODEL_NAME=ft_formular_swinv2_vit_handWrite

export SAVE_PATH=/home/public/yushilin/formular/result/okay/${MODEL_NAME}
export LOG_DIR=log_${MODEL_NAME}
#export DATA=/home/public/yushilin/formular/harvard/data/
#DATA=/home/public/yushilin/formular/taojuan/
DATA=/home/public/yushilin/handwrite/HandWriteFormula/
mkdir ${LOG_DIR}
export BSZ=38
export valid_BSZ=38



CUDA_VISIBLE_DEVICES=0,1 python $(which fairseq-train) --data-type formularHandWrite --input-size 96-384 --user-dir ./ --task text_recognition \
            --arch swinv2_tiny_patch4_window8  --seed 1111 --optimizer adam --lr 5e-05 --lr-scheduler inverse_sqrt \
            --warmup-init-lr 1e-8 --warmup-updates 500 --weight-decay 0.0001 --log-format tqdm  --log-interval 10 \
            --batch-size ${BSZ} --batch-size-valid ${valid_BSZ} --save-dir ${SAVE_PATH} --tensorboard-logdir ${LOG_DIR} \
            --max-epoch 300 --patience 20 --ddp-backend legacy_ddp --num-workers 4 --preprocess Syn --update-freq 1 \
            --skip-invalid-size-inputs-valid-test \
            --adapt-dictionary \
            --log-interval 1500 \
            --dict dictionary/handWriteFormulaComp.txt \
            --encoder-pretrained-url /home/public/yushilin/ocr/model/pretrained/encoder/swinv2_tiny_patch4_window8_256.pth \
            --data ${DATA} 
