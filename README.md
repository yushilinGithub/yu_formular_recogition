# OKayOCR  image->Latex

## Introduction
Extend Trocr (https://github.com/microsoft/unilm/tree/master/trocr) for formular recognization, convert image to Latex code, I have tried original model using Vit or beit as encoder, and unilm as decoder, I found it doesn't deal well with charactor size variation,  then I add irpe(image relative position encoding) (https://arxiv.org/abs/2107.14222) to beit. finally I think Swin Transformer may  can fit this project well, So I changed encoder to Swin Transformer, It bring me very good result, bleu4 to 88.93. I guess if I have more time to train im2latex-100k dataset, I can get start-of-art result, I need to use our data to get there, get start-of-art result isn't our purpose. you can use this project to train ocr as well.

# download im2latex-100k dataset.
```
cd data
wget http://lstm.seas.harvard.edu/latex/data/im2latex_validate_filter.lst
wget http://lstm.seas.harvard.edu/latex/data/im2latex_train_filter.lst
wget http://lstm.seas.harvard.edu/latex/data/im2latex_test_filter.lst
wget http://lstm.seas.harvard.edu/latex/data/formula_images_processed.tar.gz
wget http://lstm.seas.harvard.edu/latex/data/im2latex_formulas.norm.lst
tar -zxvf formula_images_processed.tar.gz
```
# down load pretrained model 
  the model is located at  https://github.com/microsoft/Swin-Transformer, by far I trained tiny-model as encoder, and In my experimence, I found that pretrained decoder  model seem like didn't facilatite the result.
# dependance
  pip install -r requirement.txt
# run the code
    bash run_formular.sh
# eval
    bash eval.sh