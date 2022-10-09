# formular-recognition

## Introduction
Original repositary is located at [TrOCR](https://github.com/microsoft/unilm/tree/master/trocr), you can use pretrained moded from trocr, thanks for the original one.
 [TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models](https://arxiv.org/abs/2109.10282), Minghao Li, Tengchao Lv, Lei Cui, Yijuan Lu, Dinei Florencio, Cha Zhang, Zhoujun Li, Furu Wei, Preprint 2021. I also tried TrOCR, it is not good as this one.

- vit, deit + transformer decoder (original one)
- image relative positional encoding + transformer decoder [Rethinking and Improving Relative Position Encoding for Vision Transformer](https://arxiv.org/abs/2107.14222)
- swin transformer + transformer decoder [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)
- Cswin transformer + transformer decoder [CSWin Transformer: A General Vision Transformer Backbone with Cross-Shaped Windows](https://arxiv.org/abs/2107.00652)
- swin transformer v2 + vit + transformer (best) [Swin Transformer V2: Scaling Up Capacity and Resolution](https://arxiv.org/abs/2111.09883)

 

The TrOCR is currently implemented with the fairseq library. We hope to convert the models to the Huggingface format later.


# Installation
## Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.5.0
* Python version >= 3.6
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* [fairseq](https://github.com/facebookresearch/fairseq)

### To install **fairseq** and develop locally:
``` bash
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
```
### To install the latest stable release version of **fairseq** (0.10.x)

```
pip install fairseq
```
~~~bash
pip install pybind11
pip install -r requirements.txt
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" 'git+https://github.com/NVIDIA/apex.git'
~~~

## Usage

# download im2latex-100k dataset.
```
cd data
wget https://zenodo.org/record/56198/files/formula_images.tar.gz
wget https://zenodo.org/record/56198/files/im2latex_train.lst
wget https://zenodo.org/record/56198/files/im2latex_test.lst
wget https://zenodo.org/record/56198/files/im2latex_validate.lst
tar -zxvf formula_images.tar.gz
```
# preprocess data
The images in the dataset contain a LaTeX formula rendered on a full page. To accelerate training, we need to preprocess the images.
```
python scripts/preprocessing/preprocess_images.py --input-dir data/sample/images --output-dir data/sample/images_processed
```
The above command will crop the formula area, and group images of similar sizes to facilitate batching.

Next, the LaTeX formulas need to be tokenized or normalized. The code is tested for python2


```
python2 scripts/preprocessing/preprocess_formulas.py --mode normalize --input-file data/sample/im2latex_formulas.lst --output-file data/sample/im2latex_formulas.norm.lst
```

The above command will normalize the formulas. Note that this command will produce some error messages since some formulas cannot be parsed by the KaTeX parser.

Then we need to prepare train, validation and test files. We will exclude large images from training and validation set, and we also ignore formulas with too many tokens or formulas with grammar errors.

```
python scripts/preprocessing/preprocess_filter.py --filter --image-dir data/sample/images_processed --label-path data/sample/im2latex_formulas.norm.lst--data-path data/sample/im2latex_train.lst --output-path data/sample/im2latex_train_filter.lst
```

```
python scripts/preprocessing/preprocess_filter.py --filter --image-dir data/sample/images_processed --label-path data/sample/im2latex_formulas.norm.lst --data-path data/sample/im2latex_validate.lst --output-path data/sample/im2latex_validation_filter.lst
```

```
python scripts/preprocessing/preprocess_filter.py --no-filter --image-dir data/sample/images_processed --label-path data/sample/im2latex_formulas.norm.lst --data-path data/sample/im2latex_test.lst  --output-path data/sample/im2latex_test_filter.lst 
```

Finally, we generate the vocabulary from training set. All tokens occuring less than (including) 1 time will be excluded from the vocabulary.
```
python scripts/preprocessing/generate_latex_vocab.py --data-path data/sample/train_filter.lst --label-path data/sample/formulas.norm.lst --output-file data/sample/latex_vocab.txt
```
# down load pretrained model 
  the model is located at  https://github.com/microsoft/Swin-Transformer, by far I trained the tiny-model as encoder, I found that the pretrained decoder seems like didn't facilatite the result.
# create dictionary
please read the code and change it to fit your project
  ```
  cd okayDataProcess
  python generate_vocab.py --data-path [] --output-file []
  ```
# dependance
  pip install -r requirement.txt
# run the code
  bash run_command/run_capture_formula.sh
# eval
  bash run_command/eval_capture_formula.sh
