export MODEL_NAME=ft_formular_harDict_192_384_irpe_newsize
export SAVE_PATH=/home/public/yushilin/formular/harvard/${MODEL_NAME}
export LOG_DIR=log_${MODEL_NAME}
export DATA=/home/public/yushilin/formular/harvard/data
mkdir ${LOG_DIR}
export BSZ=32
export valid_BSZ=32
#--input-size 192-768
# CUDA_VISIBLE_DEVICES=1 python $(which fairseq-train) --data-type formular --input-size 192-768 --user-dir ./ --task text_recognition  --arch deit_formular  --seed 1111 --optimizer adam --lr 5e-05 --lr-scheduler inverse_sqrt \
#     --warmup-init-lr 1e-8 --warmup-updates 500 --weight-decay 0.0001 --log-format tqdm  \
#     --log-interval 10 --batch-size ${BSZ} --batch-size-valid ${valid_BSZ} --save-dir ${SAVE_PATH} \
#     --tensorboard-logdir ${LOG_DIR} --max-epoch 300 --patience 20 --ddp-backend legacy_ddp \
#     --num-workers 8 --preprocess FM --update-freq 1 \
#     --decoder-pretrained unilm\
#     --finetune-from-model /home/yushilin/workspace/ocr/unilm/trocr/pretrain/formular_small_harDict.pt --fp16 \
#     --data ${DATA} 


#deit_small_distilled_formular the differences to original one is image data augament, and change input size 384*384 to 192*768 
#deit_formular_irpe  #with irpe registered by Yushilin
CUDA_VISIBLE_DEVICES=0 python $(which fairseq-train) --data-type formular --input-size 192-768 --user-dir ./ --task text_recognition  --arch deit_formular_irpe  --seed 1111 --optimizer adam --lr 5e-05 --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-8 --warmup-updates 500 --weight-decay 0.0001 --log-format tqdm  \
    --log-interval 10 --batch-size ${BSZ} --batch-size-valid ${valid_BSZ} --save-dir ${SAVE_PATH} \
    --tensorboard-logdir ${LOG_DIR} --max-epoch 300 --patience 20 --ddp-backend legacy_ddp \
    --num-workers 8 --preprocess FM --update-freq 1 \
    --decoder-pretrained unilm\
    --finetune-from-model /home/yushilin/workspace/ocr/unilm/trocr/pretrain/formular_small_harDict.pt --fp16 \
    --data ${DATA} 
