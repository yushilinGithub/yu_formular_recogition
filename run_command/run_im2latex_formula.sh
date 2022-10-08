export MODEL_NAME=ft_formular_1m2latex_100k_longest_512
export SAVE_PATH=/home/public/yushilin/formular/result/okay/${MODEL_NAME}
export LOG_DIR=log_${MODEL_NAME}

DATA=/home/public/yushilin/formular/im2latex-100k
mkdir ${LOG_DIR}
export BSZ=48
export valid_BSZ=48

CUDA_VISIBLE_DEVICES=0,1 python $(which fairseq-train) --data-type formular --input-size 96-384 --user-dir ./ --task text_recognition  \
            --arch swinv2_tiny_patch4_window8  --seed 1111 --optimizer adam --lr 5e-05 --lr-scheduler inverse_sqrt \
            --warmup-init-lr 1e-8 --warmup-updates 500 --weight-decay 0.0001 --log-format tqdm  --log-interval 10 \
            --batch-size ${BSZ} --batch-size-valid ${valid_BSZ} --save-dir ${SAVE_PATH} --tensorboard-logdir ${LOG_DIR} \
            --max-epoch 300 --patience 20 --ddp-backend legacy_ddp --num-workers 4 --preprocess Syn --update-freq 1 \
            --skip-invalid-size-inputs-valid-test \
            --adapt-dictionary \
            --dict dictionary/im2latex_100k.txt \
            --encoder-pretrained-url /home/public/yushilin/ocr/model/pretrained/encoder/swinv2_tiny_patch4_window8_256.pth \
            --data ${DATA} 
