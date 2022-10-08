from asyncore import read
from turtle import back
from click import argument
from cv2 import merge
from matplotlib import image
import numpy as np
from PIL import Image
import argparse
import os
import glob
import random
import cv2
import albumentations as alb
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

def read_lst(filename,image_dir):
    with open(filename,"r") as f:
        data = f.readlines()
    filelist = [os.path.join(image_dir,imagePath.split(" ")[0]) for imagePath in data]
    return filelist

def getforeAugment():
    aug = alb.Compose([
                alb.RGBShift(r_shift_limit=15, g_shift_limit=15,
                           b_shift_limit=15, p=0.3),
                alb.RandomToneCurve(0.3),
                alb.GaussNoise(10, p=.6),
                alb.RandomBrightnessContrast([-.4,0.05], [-0.2,.4], True, p=0.8),
                alb.ImageCompression(95, p=.3),
                ])
    return aug
def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)
    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))
    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distored_image.reshape(image.shape)

def randomGeo(artiImage):
    height,width = artiImage.shape[:2]
    height_variation = int(height/9)
    width_vatiation = int(width/9)

    y0,x0 = np.random.randint(height_variation),np.random.randint(width_vatiation)
    y1,x1=np.random.randint(height_variation),width+np.random.randint(-width_vatiation,width_vatiation)
    y2,x2=height+np.random.randint(-height_variation,height_variation),np.random.randint(height_variation)
    y3,x3=height+np.random.randint(-height_variation,height_variation),width+np.random.randint(-width_vatiation,width_vatiation)
   


    pts1 = np.float32([[0,0],[width,0],[0,height],[width,height]])
    pts2 = np.float32([[x0,y0],[x1,y1],[x2,y2],[x3,y3]])

    M = cv2.getPerspectiveTransform(pts1,pts2)
    dstHeight = max(y2,y3)-min(y0,y1) 
    dstWidth = max(x1,x3)-min(x0,x2)
    dstA = cv2.warpPerspective(artiImage,M,(dstWidth,dstHeight),borderValue=(255,255,255))
    return dstA
def merge(args,foreList,backgroundList):
    #blend foreground image to random selected background image
    aug = getforeAugment()
    for foreFile in foreList:
        foreground = cv2.imread(foreFile)
        if random.random()>0.5:
            foreground = randomGeo(np.array(foreground))
        foreground = Image.fromarray(foreground)
        backgroundFile = random.choice(backgroundList)
        background = Image.open(backgroundFile)
        if foreground.mode != background.mode:
            foreground = foreground.convert(background.mode)
        fwidth,fheight = foreground.size
        bwidth,bheight = background.size
        if bwidth>fwidth and bheight>fheight:
            top = random.randint(0,bheight-fheight)
            left = random.randint(0,bwidth-fwidth)
            background = background.crop((left, top, left+fwidth, top+fheight))
        else:
            background = background.resize(foreground.size)
        alpha = random.uniform(0.4,0.7)
        merged = Image.blend(background, foreground,alpha)
        if random.random()>0.6:
            merged = elastic_transform(np.array(merged), alpha=300, sigma=8)

        merged = aug(image = np.array(merged))
        merged = Image.fromarray(merged["image"])
        outputPath = os.path.join(args.output,os.path.basename(foreFile))
        merged.save(outputPath)
def main(args):
    if not os.path.isdir(args.output):
        print("{} not exist, create it".format(args.output))
        os.mkdir(args.output)
    train = os.path.join(args.front_path,"im2latex_train_filter.lst") 
    test = os.path.join(args.front_path,"im2latex_test_filter.lst")
    validation = os.path.join(args.front_path,"im2latex_validation_filter.lst")
    image_dir = os.path.join(args.front_path,"formula_images_processed")
    train_file_list = read_lst(train,image_dir)
    val_file_list = read_lst(validation,image_dir)
    test_file_list = read_lst(test,image_dir)
    backgroundList = glob.glob(os.path.join(args.backgrond_path+"*.jpg"))
    merge(args,train_file_list,backgroundList)
    merge(args,val_file_list,backgroundList)
    merge(args,test_file_list,backgroundList)
    print("merge image Finished")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--front_path",type=str,default="/home/public/yushilin/formular/taojuan/")
    parser.add_argument("--backgrond_path",type=str,default="/home/public/yushilin/formular/back_ground/")
    parser.add_argument("--output",type=str,default="/home/public/yushilin/formular/mopai/simulatedData/")
    args = parser.parse_args()
    main(args)