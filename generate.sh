export MODEL=/home/public/yushilin/formular/result/okay/ft_formular_swin_taojuan/checkpoint_last.pt
export RESULT_PATH=/home/public/yushilin/formular/results/taojuan_test
#export DATA=/home/public/yushilin/formular/harvard/data/
export DATA=/home/public/yushilin/formular/taojuan/test_image_with_no_label


#swin_tiny_patch4_window7
CUDA_VISIBLE_DEVICES=1 $(which fairseq-interactive)  \
        --data-type formular  --user-dir ./ --task text_recognition --input-size 224-672 \
        --beam 5 --gen-subset test  \
        --path ${MODEL} --results-path ${RESULT_PATH} \
        --dict-path-or-url dictionary/vocab_taojuan.txt  \
        --data ${DATA}
