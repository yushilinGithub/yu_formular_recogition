
import task
import deit
import deit_models
import torch
import fairseq
from fairseq import utils
from fairseq_cli import generate

import time
from PIL import Image
import cv2
import numpy as np
import albumentations as alb
from albumentations.pytorch import ToTensorV2
from models import okayocr_model  #load to register the model
import glob
import numpy as nps
class ResizePad(object):
    def __init__(self, img_size):
        self.img_size=img_size
    def __call__(self,image):  
        old_size = image.shape[:2]  
        ratio = min(float(self.img_size[0]-10)/old_size[0],float(self.img_size[1]-10)/old_size[1]) 
        height = int(old_size[0]*ratio)
        width = int(old_size[1]*ratio)
        im = cv2.resize(image,(width,height),cv2.INTER_AREA)
        new_im = np.zeros((self.img_size[0],self.img_size[1],3),dtype=np.uint8)+255
        new_im[5:5+height,5:5+width]=im


        return new_im

from PIL import Image
import torchvision.transforms as transforms


def init(model_path, beam=5):
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [model_path],
        arg_overrides={"beam": beam, "task": "text_recognition", "data": "", "fp16": False})

        #, "arch":"swin_tiny_patch4_window7","dict":"dictionary/mopai_chinese_support_TAL_mopai_formula.txt"})
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model[0].to(device)

    img_transform = alb.Compose(
        [
            alb.ToGray(always_apply=True),
            alb.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738)),
            #alb.Sharpen()
            ToTensorV2(),
        ]
    )


    device = "cuda" if torch.cuda.is_available() else "cpu"
    model[0].to(device)

    img_transform = transforms.Compose([
        transforms.Resize((384, 384), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])


    generator = task.build_generator(
        model, cfg.generation, extra_gen_cls_kwargs={'lm_model': None, 'lm_weight': None}
    )

    bpe = task.build_bpe(cfg.bpe)

    return model, cfg, task, generator, bpe, img_transform, device


def preprocess(img_path, img_transform):

    im = cv2.imread(img_path)#.convert('RGB').resize((384, 384))
    resize_pad = ResizePad([96,384])
    im = resize_pad(im)
    im = img_transform(image=im)["image"].unsqueeze(0).to(device).float()

    sample = {
        'net_input': {"imgs": im},
    }

    return sample


def get_text(cfg, generator, model, sample, bpe):
    decoder_output = task.inference_step(generator, model, sample, prefix_tokens=None, constraints=None)
    decoder_output = decoder_output[0][0]       #top1

    hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
        hypo_tokens=decoder_output["tokens"].int().cpu(),
        src_str="",
        alignment=decoder_output["alignment"],
        align_dict=None,
        tgt_dict=model[0].decoder.dictionary,
        remove_bpe=cfg.common_eval.post_process,
        extra_symbols_to_ignore=generate.get_symbols_to_strip_from_output(generator),
    )

    #detok_hypo_str = bpe.decode(hypo_str)
    return hypo_str


if __name__ == '__main__':
    import os
    #model_path = '/home/public/yushilin/formular/mopai/model/chinese_mopai_112_336_TAL_203.pt'
    model_path = "/home/public/yushilin/ocr/model/handWriteFormula/swinv2_vit_best.pt"
    #model_path = "/home/public/yushilin/formular/result/okay/ft_formular_swin_chinese_support/207_taojuan_chinese_support.pt"
    jpg_path = "/home/public/yushilin/handwrite/HandWriteFormula/test/Task1/images/"
    #jpg_path = "/home/public/yushilin/formular/taojuan_chinese_support_test/"
    imagePath = glob.glob(jpg_path+"*.png")
    beam = 10
    model, cfg, task, generator, bpe, img_transform, device = init(model_path, beam)
    times = []
    lines = []
    for imagep in imagePath:
        print(imagep)
        sample = preprocess(imagep, img_transform)
        time1 = time.time()
        text = get_text(cfg, generator, model, sample, bpe)
        time2 = time.time()
        time_usage = time2-time1
        print("time_usage",time_usage)
        times.append(time_usage)

        lines.append(os.path.basename(imagep)+"\t"+text)
        
    with open("test2.txt","w",encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("mean time",np.mean(time_usage))

import task
import deit
import deit_models
import torch
import fairseq
from fairseq import utils
from fairseq_cli import generate
from PIL import Image
import torchvision.transforms as transforms


def init(model_path, beam=5):
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [model_path],
        arg_overrides={"beam": beam, "task": "text_recognition", "data": "", "fp16": False})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model[0].to(device)

    img_transform = transforms.Compose([
        transforms.Resize((384, 384), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    generator = task.build_generator(
        model, cfg.generation, extra_gen_cls_kwargs={'lm_model': None, 'lm_weight': None}
    )

    bpe = task.build_bpe(cfg.bpe)

    return model, cfg, task, generator, bpe, img_transform, device


def preprocess(img_path, img_transform):
    im = Image.open(img_path).convert('RGB').resize((384, 384))
    im = img_transform(im).unsqueeze(0).to(device).float()

    sample = {
        'net_input': {"imgs": im},
    }

    return sample


def get_text(cfg, generator, model, sample, bpe):
    decoder_output = task.inference_step(generator, model, sample, prefix_tokens=None, constraints=None)
    decoder_output = decoder_output[0][0]       #top1

    hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
        hypo_tokens=decoder_output["tokens"].int().cpu(),
        src_str="",
        alignment=decoder_output["alignment"],
        align_dict=None,
        tgt_dict=model[0].decoder.dictionary,
        remove_bpe=cfg.common_eval.post_process,
        extra_symbols_to_ignore=generate.get_symbols_to_strip_from_output(generator),
    )


    detok_hypo_str = bpe.decode(hypo_str)

    return detok_hypo_str


if __name__ == '__main__':
    model_path = 'path/to/model'
    jpg_path = "path/to/pic"
    beam = 5

    model, cfg, task, generator, bpe, img_transform, device = init(model_path, beam)

    sample = preprocess(jpg_path, img_transform)

    text = get_text(cfg, generator, model, sample, bpe)
 
    print(text)

    print('done')


