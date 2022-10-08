#对标注的数据批量前向


# __dir__ = os.path.dirname(os.path.abspath(__file__))

# sys.path.append(__dir__)
# sys.path.append(os.path.abspath(os.path.join(__dir__, '..'))) 
import sys
sys.path.append("../..")

import task
import os
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
import argparse
import tqdm
import json
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
def init(model_path, beam=5):
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [model_path],
        arg_overrides={"beam": beam, "task": "text_recognition", "data": "", "fp16": False, "arch":"swin_tiny_patch4_window7"})
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
    generator = task.build_generator(
        model, cfg.generation, extra_gen_cls_kwargs={'lm_model': None, 'lm_weight': None}
    )
    bpe = task.build_bpe(cfg.bpe)
    return model, cfg, task, generator, bpe, img_transform, device

def preprocess(im, img_transform,device):
    #.convert('RGB').resize((384, 384))
    resize_pad = ResizePad([224,672])
    im = resize_pad(im)
    im = img_transform(image=im)["image"].unsqueeze(0).to(device).float()
    sample = {
        'net_input': {"imgs": im},
    }
    return sample

def get_text(cfg, generator, model, sample,task,bpe):
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
def process(detList,model,cfg,task,generator,bpe,img_transform,device,imageFolder,split):
    data = {ins["image_name"]:[] for ins in detList}
    for instance in tqdm.tqdm(detList):
        subImageName =os.path.join(imageFolder,str(instance["image_id"])+"_"+str(instance["id"])+split+".jpg")
        imageName = instance["image_name"]
        if not os.path.exists(subImageName):
            continue
        im = cv2.imread(subImageName)
    
        sample = preprocess(im, img_transform,device)
        text = get_text(cfg, generator, model, sample, task,bpe)
        x,y,w,h = instance['bbox']

        result = {"transcription":text,"points":[[x,y],[x+w,y],[x+w,y+h],[x,y+h]],"difficult": False}
        data[imageName].append(result)
    return data
def saveLabel(args,dict,split):
    txt = ""
    for imageName,value in dict.items():
        txt+="{}2017/".format(split)+imageName+"\t"+str(value)+"\n"
    with open(os.path.join(args.desFolder,"{}.txt".format(split)),"w") as f:
        f.write(txt)
def main(args):
    imageFolder = os.path.join(args.srcFolder,"images")
    trainDet = os.path.join(args.srcFolder,"train.json")
    valDet = os.path.join(args.srcFolder,"val.json")
    testDet = os.path.join(args.srcFolder,"test.json")
    with open(trainDet,"r") as f:
        trainDet = json.load(f)
    with open(valDet,"r") as f:
        valDet = json.load(f)
    with open(testDet,"r") as f:
        testDet = json.load(f)
    model, cfg, task, generator, bpe, img_transform, device = init(args.modelPath, beam=5)
    trainLabel = process(trainDet,model,cfg,task,generator,bpe,img_transform,device,imageFolder,"train")
    #valLabel = process(valDet,model,cfg,task,generator,bpe,img_transform,device,imageFolder,"val")
    #testLabel = process(testDet,model,cfg,task,generator,bpe,img_transform,device,imageFolder,"test")

    saveLabel(args,trainLabel,"train_exp")
    #saveLabel(args,valLabel,"val")
    #saveLabel(args,testLabel,"train")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--srcFolder",type=str,default="/home/public/yushilin/formular/mopai/realData/")
    parser.add_argument("--desFolder",type=str,default="/home/public/yushilin/formular/mopai/realData/")
    parser.add_argument("--modelPath",type=str,default='/home/public/yushilin/formular/result/okay/ft_formular_swin_chinese_support/207_mopai_chinese_support.pt')
    args = parser.parse_args()
    main(args)