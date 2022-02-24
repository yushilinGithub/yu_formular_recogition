import torchvision.transforms as transforms
# from torchvision.transforms.functional import InterpolationMode
from PIL import Image, ImageFilter
import random
import torch
import numpy as np
import logging
from enum import Enum
import albumentations as alb
from albumentations.pytorch import ToTensorV2

# 0: InterpolationMode.NEAREST,
# 2: InterpolationMode.BILINEAR,
# 3: InterpolationMode.BICUBIC,
# 4: InterpolationMode.BOX,
# 5: InterpolationMode.HAMMING,
# 1: InterpolationMode.LANCZOS,
class InterpolationMode():
    NEAREST = 0
    BILINEAR = 2
    BICUBIC = 3
    BOX = 4
    HAMMING = 5
    LANCZOS = 1

logger = logging.getLogger(__name__)

class ResizePad(object):

    def __init__(self, imgH=64, imgW=3072, keep_ratio_with_pad=True):
        self.imgH = imgH
        self.imgW = imgW
        assert keep_ratio_with_pad == True
        self.keep_ratio_with_pad = keep_ratio_with_pad

    def __call__(self, im):        

        old_size = im.size  # old_size[0] is in (width, height) format

        ratio = min(float(self.imgH)/old_size[1],float(self.imgW)/old_size[0])
        new_size = tuple([int(x*ratio) for x in old_size])
        im = im.resize(new_size, Image.BICUBIC)

        new_im = Image.new("RGB", (self.imgW, self.imgH))
        new_im.paste(im, (0, 0))

        return new_im

class ResizePad_array(object):

    def __init__(self, imgH=64, imgW=3072, keep_ratio_with_pad=True):
        self.imgH = imgH
        self.imgW = imgW
        assert keep_ratio_with_pad == True
        self.keep_ratio_with_pad = keep_ratio_with_pad

    def __call__(self,image):        

        old_size = image.shape[:2]  

        ratio = min(float(self.imgH)/old_size[0],float(self.imgW)/old_size[1])
        height = int(old_size[0]*ratio)
        width = int(old_size[1]*ratio)

        im = cv2.resize(image,(width,height),cv2.INTER_AREA)

        new_im = np.zeros((self.imgH,self.imgW,3),dtype=np.uint8)+255
        new_im[0:height,0:width]=im

        return new_im


class WeightedRandomChoice:

    def __init__(self, trans, weights=None):
        self.trans = trans
        if not weights:
            self.weights = [1] * len(trans)
        else:
            assert len(trans) == len(weights)
            self.weights = weights

    def __call__(self, img):
        t = random.choices(self.trans, weights=self.weights, k=1)[0]
        try:
            tfm_img = t(img)
        except Exception as e:
            logger.warning('Error during data_aug: '+str(e))
            return img

        return tfm_img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class Dilation(torch.nn.Module):

    def __init__(self, kernel=3):
        super().__init__()
        self.kernel=kernel

    def forward(self, img):
        return img.filter(ImageFilter.MaxFilter(self.kernel))

    def __repr__(self):
        return self.__class__.__name__ + '(kernel={})'.format(self.kernel)

class Erosion(torch.nn.Module):

    def __init__(self, kernel=3):
        super().__init__()
        self.kernel=kernel

    def forward(self, img):
        return img.filter(ImageFilter.MinFilter(self.kernel))

    def __repr__(self):
        return self.__class__.__name__ + '(kernel={})'.format(self.kernel)

class Underline(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, img):
        img_np = np.array(img.convert('L'))
        black_pixels = np.where(img_np < 50)
        try:
            y1 = max(black_pixels[0])
            x0 = min(black_pixels[1])
            x1 = max(black_pixels[1])
        except:
            return img
        for x in range(x0, x1):
            for y in range(y1, y1-3, -1):
                try:
                    img.putpixel((x, y), (0, 0, 0))
                except:
                    continue
        return img

class KeepOriginal(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        return img    


test_transform = alb.Compose(
    [
        alb.ToGray(always_apply=True),
        alb.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738)),
        # alb.Sharpen()
        ToTensorV2(),
    ]
)

def build_formular_aug(mode):
    if mode=="train":
        train_transform = alb.Compose(
            [
                alb.Compose(
                    [alb.ShiftScaleRotate(shift_limit=0, scale_limit=(-.25, 0), rotate_limit=1, border_mode=0, interpolation=3,
                                        value=[255, 255, 255], p=1),
                    alb.GridDistortion(distort_limit=0.1, border_mode=0, interpolation=3, value=[255, 255, 255], p=.5)], p=.15),
                # alb.InvertImg(p=.15),
                alb.RGBShift(r_shift_limit=15, g_shift_limit=15,
                            b_shift_limit=15, p=0.3),
                alb.GaussNoise(10, p=.2),
                alb.RandomBrightnessContrast(.05, (-.2, 0), True, p=0.2),
                alb.ImageCompression(95, p=.3),
                alb.ToGray(always_apply=True),
                #alb.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738)),
                # alb.Sharpen()
                #ToTensorV2(),
            ]
        )
        return train_transform
    else:
        test_transform = alb.Compose(
            [
                alb.ToGray(always_apply=True),
                alb.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738)),
                # alb.Sharpen()
                ToTensorV2(),
            ]
        )
        return test_transform


def build_data_aug(size, mode, resnet=False, resizepad=False):
    if resnet:
        norm_tfm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        norm_tfm = transforms.Normalize(0.5, 0.5)
    if resizepad:
        resize_tfm = ResizePad(imgH=size[0], imgW=size[1])
    else:
        resize_tfm = transforms.Resize(size, interpolation=InterpolationMode.HAMMING)
    if mode == 'train':
        return transforms.Compose([
            WeightedRandomChoice([
                # transforms.RandomHorizontalFlip(p=1),
                transforms.RandomRotation(degrees=(-10, 10), expand=True, fill=255),
                transforms.GaussianBlur(3),
                Dilation(3),
                Erosion(3),
                transforms.Resize((size[0] // 3, size[1] // 3), interpolation=InterpolationMode.HAMMING),
                Underline(),
                KeepOriginal(),
            ]),
            resize_tfm,
            transforms.ToTensor(),
            norm_tfm
        ])
    else:
        return transforms.Compose([
            resize_tfm,
            transforms.ToTensor(),
            norm_tfm
        ])

import os
import cv2
if __name__ == '__main__':
    image_size = (192,768)
    #tfm = build_data_aug(image_size, mode='train', resizepad=True)  
    tfm = build_formular_aug(image_size, mode='train')
    resizepad = ResizePad_array(192,768)
    dir = "test"
    for image in os.listdir(dir):
        img = cv2.imread(os.path.join(dir,image))
        img = resizepad(img)
        img = tfm(image=img)
        #image.save(os.path.join('result',image))
        cv2.imwrite(os.path.join('result',image),img['image'])