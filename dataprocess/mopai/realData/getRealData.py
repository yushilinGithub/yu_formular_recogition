from bitarray import test
from builtins import type
from email.policy import default
from ast import arg
import numpy as np
import pandas as pd
import argparse
import os
import json
import cv2

def get_data(args):
    train_instances = os.path.join(args.src_path,"annotations/instances_train2017.json")
    val_instances = os.path.join(args.src_path,"annotations/instances_val2017.json")
    test_instances = os.path.join(args.src_path,"annotations/instances_test2017.json")
    train_dir = os.path.join(args.src_path,"train2017")
    test_dir = os.path.join(args.src_path,"test2017")
    val_dir = os.path.join(args.src_path,"val2017")
    with open(train_instances,"r") as f:
        train_annotation = json.load(f)
    with open(val_instances,"r") as f:
        val_annotation = json.load(f)
    with open(test_instances,"r") as f:
        test_annotation = json.load(f)
    return train_annotation,val_annotation,test_annotation,train_dir,val_dir,test_dir
def parserAnnotation(annotation):
    #id,image_id,image_name,box
    boxInstance = []
    id_to_name = {instance["id"]:instance["file_name"] for instance in annotation["images"]}
    for box in annotation["annotations"]:
        boxInstance.append({"id":box["id"],"image_id":box["image_id"],"image_name":id_to_name[box["image_id"]],"bbox":box["bbox"]})
    return boxInstance
def get_subImage(dir,annotaions,dstdir,folder):
    for ann in annotaions:
        imagePath = os.path.join(dir,ann["image_name"])
        image = cv2.imread(imagePath)
        x,y,w,h = ann["bbox"]
        subImage = image[y:y+h,x:x+w]
        dstPath = os.path.join(dstdir,str(ann["image_id"])+"_"+str(ann["id"])+folder+".jpg")
        if subImage.shape[0]>10 and subImage.shape[1]>10:
            cv2.imwrite(dstPath,subImage)
def main(args):
    train_annotation,val_annotation,test_annotation,train_dir,val_dir,test_dir = get_data(args)
    train_annotation = parserAnnotation(train_annotation)
    val_annotation = parserAnnotation(val_annotation)
    test_annotation = parserAnnotation(test_annotation)
    get_subImage(train_dir,train_annotation,args.dst_path,"train")
    get_subImage(val_dir,val_annotation,args.dst_path,"val")
    get_subImage(test_dir,test_annotation,args.dst_path,"test")
    with open(os.path.join(args.dst_path,"train.json"),"w") as f:
        json.dump(train_annotation,f)
    with open(os.path.join(args.dst_path,"val.json"),"w") as f:
        json.dump(val_annotation,f)
    with open(os.path.join(args.dst_path,"test.json"),"w") as f:
        json.dump(test_annotation,f)
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path",type=str,default="/home/public/dataset/ques_formula_12345_hx/")
    parser.add_argument("--dst_path",type=str,default="/home/public/yushilin/formular/mopai/realData/")
    args = parser.parse_args()
    main(args)