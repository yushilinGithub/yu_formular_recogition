from platform import libc_ver
from builtins import isinstance
import glob
import logging
import os

import random

from webbrowser import BackgroundBrowser
import cv2
from cv2 import erode
from sklearn.preprocessing import binarize

import torch
from fairseq.data import FairseqDataset, data_utils
from natsort import natsorted
from PIL import Image
from tqdm import tqdm
import numpy as np
import glob 

from albumentations.pytorch import ToTensorV2

logger = logging.getLogger(__name__)


def default_collater(target_dict, samples, dataset=None):
    if not samples:
        return None
    if any([sample is None for sample in samples]):
        if not dataset:
            return None
        len_batch = len(samples)        
        while True:
            samples.append(dataset[random.choice(range(len(dataset)))])
            samples =list(filter (lambda x:x is not None, samples))
            if len(samples) == len_batch:
                break        
    indices = []

    imgs = [] # bs, c, h , w
    target_samples = []
    target_ntokens = 0

    for sample in samples:
        index = sample['id']
        indices.append(index)

        
        imgs.append(sample['tfm_img'])
        
        target_samples.append(sample['label_ids'].long())
        target_ntokens += len(sample['label_ids'])

    num_sentences = len(samples)

    target_batch = data_utils.collate_tokens(target_samples,
                                            pad_idx=target_dict.pad(),
                                            eos_idx=target_dict.eos(),
                                            move_eos_to_beginning=False)
    rotate_batch = data_utils.collate_tokens(target_samples,
                                            pad_idx=target_dict.pad(),
                                            eos_idx=target_dict.eos(),
                                            move_eos_to_beginning=True)                                               

    indices = torch.tensor(indices, dtype=torch.long)
    imgs = torch.stack(imgs, dim=0)

    return {
        'id': indices,
        'net_input': {
            'imgs': imgs,
            'prev_output_tokens': rotate_batch
        },
        'ntokens': target_ntokens,
        'nsentences': num_sentences,            
        'target': target_batch
    }

def read_txt_and_tokenize(txt_path: str, bpe, target_dict):
    annotations = []
    with open(txt_path, 'r', encoding='utf8') as fp:
        for line in fp.readlines():
            line = line.rstrip()
            if not line:
                continue
            line_split = line.split(',', maxsplit=8)
            quadrangle = list(map(int, line_split[:8]))
            content = line_split[-1]

            if bpe:
                encoded_str = bpe.encode(content)
            else:
                encoded_str = content

            xs = [quadrangle[i] for i in range(0, 8, 2)]
            ys = [quadrangle[i] for i in range(1, 8, 2)]
            bbox = [min(xs), min(ys), max(xs), max(ys)]
            annotations.append({'bbox': bbox, 'encoded_str': encoded_str, 'category_id': 0, 'segmentation': [quadrangle]})  # 0 for text, 1 for background

    return annotations

def SROIETask2(root_dir: str, bpe, target_dict, crop_img_output_dir=None):
    data = []
    img_id = -1

    crop_data = []
    crop_img_id = -1

    image_paths = natsorted(list(glob.glob(os.path.join(root_dir, '*.jpg'))))
    for jpg_path in tqdm(image_paths):
        im = Image.open(jpg_path).convert('RGB')
        
        img_w, img_h = im.size
        img_id += 1

        txt_path = jpg_path.replace('.jpg', '.txt')
        annotations = read_txt_and_tokenize(txt_path, bpe, target_dict) 
        img_dict = {'file_name': jpg_path, 'width': img_w, 'height': img_h, 'image_id':img_id, 'annotations':annotations}
        data.append(img_dict)

        for ann in annotations:
            crop_w = ann['bbox'][2] - ann['bbox'][0]
            crop_h = ann['bbox'][3] - ann['bbox'][1]

            if not (crop_w > 0 and crop_h > 0):
                logger.warning('Error occurs during image cropping: {} has a zero area bbox.'.format(os.path.basename(jpg_path)))
                continue
            crop_img_id += 1
            crop_im = im.crop(ann['bbox'])
            if crop_img_output_dir:
                crop_im.save(os.path.join(crop_img_output_dir, '{:d}.jpg'.format(crop_img_id)))
            crop_img_dict = {'img':crop_im, 'file_name': jpg_path, 'width': crop_w, 'height': crop_h, 'image_id':crop_img_id, 'encoded_str':ann['encoded_str']}
            crop_data.append(crop_img_dict)

    return data, crop_data

class SROIETextRecognitionDataset(FairseqDataset):
    def __init__(self, root_dir, tfm, bpe_parser, target_dict, crop_img_output_dir=None):
        self.root_dir = root_dir
        self.tfm = tfm            
        self.target_dict = target_dict
        # self.bpe_parser = bpe_parser
        self.ori_data, self.data = SROIETask2(root_dir, bpe_parser, target_dict, crop_img_output_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_dict = self.data[idx]
        
        image = img_dict['img']
        encoded_str = img_dict['encoded_str']
        input_ids = self.target_dict.encode_line(encoded_str, add_if_not_exist=False)

        tfm_img = self.tfm(image)   # h, w, c
        return {'id': idx, 'tfm_img': tfm_img, 'label_ids': input_ids}

    def size(self, idx):
        img_dict = self.data[idx]

        encoded_str = img_dict['encoded_str']
        input_ids = self.target_dict.encode_line(encoded_str, add_if_not_exist=False)
        return len(input_ids)

    def num_tokens(self, idx):
        return self.size(idx)


    def collater(self, samples):
        return default_collater(self.target_dict, samples)

def STR(gt_path, bpe_parser):
    root_dir = os.path.dirname(gt_path)
    data = []
    img_id = 0
    with open(gt_path, 'r') as fp:
        for line in tqdm(list(fp.readlines()), desc='Loading STR:'):
            line = line.rstrip()
            temp = line.split('\t', 1)
            img_file = temp[0]
            text = temp[1]

            img_path = os.path.join(root_dir, 'image', img_file)  
            if not bpe_parser:
                encoded_str = text
            else:
                encoded_str = bpe_parser.encode(text)      

            data.append({'img_path': img_path, 'image_id':img_id, 'text':text, 'encoded_str':encoded_str})
            img_id += 1

    return data


class SyntheticTextRecognitionDataset(FairseqDataset):
    def __init__(self, gt_path, tfm, bpe_parser, target_dict):
        self.gt_path = gt_path
        self.tfm = tfm
        self.target_dict = target_dict
        self.data = STR(gt_path, bpe_parser)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_dict = self.data[idx]

        image = Image.open(img_dict['img_path']).convert('RGB')
        encoded_str = img_dict['encoded_str']
        input_ids = self.target_dict.encode_line(encoded_str, add_if_not_exist=False)

        tfm_img = self.tfm(image)  # h, w, c
        return {'id': idx, 'tfm_img': tfm_img, 'label_ids': input_ids}

    def size(self, idx):
        img_dict = self.data[idx]

        encoded_str = img_dict['encoded_str']
        input_ids = self.target_dict.encode_line(encoded_str, add_if_not_exist=False)
        return len(input_ids)

    def num_tokens(self, idx):
        return self.size(idx)

    def collater(self, samples):
        return default_collater(self.target_dict, samples)

class ResizePad(object):
    def __init__(self, img_size, split, syn=False):
        self.img_size=img_size
        self.split = split
        self.syn = syn
    def cut_white_space(self,image):
        gray =cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _,thresh = cv2.threshold(gray,200,255,cv2.THRESH_BINARY_INV)

        coords = cv2.findNonZero(thresh)  # Find all non-zero points (text)
        if isinstance(coords,np.ndarray) or isinstance(coords,list):
            a, b, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
            image = image[b:b+h, a:a+w]
        return image

    def __call__(self,image,cut_white_space=True):   #有些白底黑字的图片太大，需要对其进行剪切

        if cut_white_space:
            image = self.cut_white_space(image)
        old_size = image.shape[:2]  

        ratio = min(float(self.img_size[0]-10)/old_size[0],float(self.img_size[1]-10)/old_size[1]) 


        if self.split !="train":   #如果不是训练，将图片放在右上角

            height = int(old_size[0]*ratio)
            width = int(old_size[1]*ratio)
            im = cv2.resize(image,(width,height),cv2.INTER_AREA)
            new_im = np.zeros((self.img_size[0],self.img_size[1],3),dtype=np.uint8)+255
            new_im[5:5+height,5:5+width]=im
        else:

            if not self.syn: 
                ratio = random.uniform(0.75,1)*ratio

            height = int(old_size[0]*ratio)
            width = int(old_size[1]*ratio)
            im = cv2.resize(image,(width,height),cv2.INTER_AREA)
            new_im = np.zeros((self.img_size[0],self.img_size[1],3),dtype=np.uint8)+255
            y_start = random.randint(0,self.img_size[0]-height)
            x_start = random.randint(1,self.img_size[1]-width)
            new_im[y_start:y_start+height,x_start:x_start+width]=im

        return new_im

def FMR(gt_path,mode,bpe_parser=None):
    normFile = "im2latex_formulas.norm.lst"
    with open(os.path.join(gt_path,normFile),"r") as f:

        latex = f.readlines()

    split_file={"train":"im2latex_train_filter.lst","test":"im2latex_test_filter.lst","valid":"im2latex_validation_filter.lst"}
    data = []
    img_id = 0
    
    with open(os.path.join(gt_path,split_file[mode]), 'r') as fp:
        for line in tqdm(list(fp), desc='Loading formular data:'):
            line = line.rstrip()
            temp = line.split(' ')
            img_file = temp[0]

            need_background = False if img_file.startswith(("val2017","test2017","train2017")) else True #真实拍摄的图片不用融合背景


            text = latex[int(temp[1])]

            img_path = os.path.join(gt_path, 'formula_images_processed', img_file)  
            if not bpe_parser:
                encoded_str = text
            else:
                encoded_str = bpe_parser.encode(text)      

            data.append({'img_path': img_path, 'image_id':img_id, 'text':text, 'encoded_str':encoded_str,"need_background":need_background})
            img_id += 1
    return data
def xunfeiHFMR(gt_path,mode):
    data = []
    image_id=0
    if mode=="train":
        subjects = ["chemistry","math","physics"]
    else:
        subjects = ["Task1"]
    for subject in subjects:
        label = os.path.join(gt_path,mode,subject,"{}.txt".format(mode))
        with open(label,"r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    splits = line.split("\t",1)
                    if len(splits)==2:
                        image_name,latex = splits
                    else:
                        image_name = splits[0]
                        latex = ""
                    image_path = os.path.join(gt_path,mode,subject,"images",image_name)
                    data.append({'img_path': image_path, 'image_id':image_id, 'encoded_str':latex,"need_background":False})
                    image_id += 1
    return data
class FormularRecognitionDataset(FairseqDataset):
    def __init__(self,gt_path,tfm,target_dict,input_size,split,data_loader="FMR"):

        self.gt_path = gt_path
        self.tfm = tfm
        self.target_dict = target_dict
        self.input_size = input_size
        self.resizePad = ResizePad(img_size = input_size,split=split)

        self.data = eval(data_loader)(gt_path,mode=split)

        if split=='train':
            random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_dict = self.data[index]
        if not os.path.exists(img_dict['img_path']):
            print("{} do not exist".format(img_dict['img_path']))
        image = cv2.imread(img_dict['img_path'])
        
        image = self.resizePad(image,cut_white_space=False) 
        encoded_str = img_dict['encoded_str']

        input_ids = self.target_dict.encode_line(encoded_str, add_if_not_exist=False)

        tfm_img = self.tfm(image=image)["image"]  # h, w, c
        return {'id': index, 'tfm_img': tfm_img, 'label_ids': input_ids}

    def size(self, idx):
        img_dict = self.data[idx]

        encoded_str = img_dict['encoded_str']
        input_ids = self.target_dict.encode_line(encoded_str, add_if_not_exist=False)
        return len(input_ids)

    def num_tokens(self, idx):
        return self.size(idx)

    def collater(self, samples):
        return default_collater(self.target_dict, samples)


def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """

    from scipy.ndimage.interpolation import map_coordinates
    from scipy.ndimage.filters import gaussian_filter
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

def rotate_image(image, angle):
    """Rotate the image counterclockwise.
    Rotate the image such that the rotated image is enclosed inside the
    tightest rectangle. The area not occupied by the pixels of the original
    image is colored black.
    Parameters
    ----------
    image : numpy.ndarray
        numpy image
    angle : float
        angle by which the image is to be rotated. Positive angle is
        counterclockwise.
    Returns
    -------
    numpy.ndarray
        Rotated Image
    """
    # get dims, find center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    image = cv2.warpAffine(image, M, (nW, nH), False,borderValue=(255,255,255))

    gray =cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = cv2.resize(image, (w,h))
    _,thresh = cv2.threshold(gray,200,255,cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(thresh)  # Find all non-zero points (text)
    if isinstance(coords,np.ndarray) or isinstance(coords,list):
        a, b, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
        image = image[b:b+h, a:a+w]
        
    return image
def randomGeo(artiImage):
    """random geometry transformation of images
    """
    height,width = artiImage.shape[:2]
    
    height_variation = int(height/6)+1
    width_vatiation = int(width/6)+1

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

class SynthesizerRealFormular(FairseqDataset):
    def __init__(self,gt_path,tfm,target_dict,input_size,split,backGroundPath=None,data_loader="FMR"):
        import albumentations as alb
        if backGroundPath:
            self.background = glob.glob(os.path.join(backGroundPath,"*.png"))+ glob.glob(os.path.join(backGroundPath,"*.jpg"))
        self.gt_path = gt_path
        self.tfm = tfm
        self.split = split
        self.target_dict = target_dict
        self.input_size = input_size
        self.resizePad = ResizePad(img_size = input_size,split=split,syn=True)
        self.data = eval(data_loader)(gt_path,mode=split)
        self.countMergeIndex = 0
        self.backgroundImage = None
        self.kernel = np.ones((3, 3), np.uint8)
        self.aug = alb.Compose([
                alb.RGBShift(r_shift_limit=15, g_shift_limit=15,
                           b_shift_limit=15, p=0.3),
                alb.RandomToneCurve(0.3),
                alb.GaussNoise(10,p=.6),
                alb.GaussianBlur(blur_limit=(3,5),p=.6),
                #alb.AdvancedBlur(noise_limit=(0.75,1,5),p=0.6),
                alb.RandomBrightnessContrast([-.4,0.05], [-0.2,.4], True, p=0.8),
                alb.ImageCompression(95, p=.3),                
                ])
        if split=="train":
            random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_dict = self.data[index]

        need_background = img_dict["need_background"]

        if not os.path.exists(img_dict['img_path']):
            print(img_dict['img_path'])
        
        image = cv2.imread(img_dict['img_path'])
        if self.split == "train":
            if random.random()>0.6:
                image = randomGeo(image)
            if random.random()>0.8:
                image = rotate_image(image,np.random.randint(-20,20))
            if random.random()>0.8:
                try:
                    if random.randint(0,1):
                        image  = cv2.erode(image, self.kernel, iterations=1)
                    else:
                        image = cv2.dilate(image, self.kernel, iterations=1)
                except:
                    pass
            if need_background and random.random()>0.1:
                image = self.mergeBackground(image)
                
            image = self.aug(image = image)["image"]

        image = self.resizePad(image,cut_white_space=False)
        encoded_str = img_dict['encoded_str']

        input_ids = self.target_dict.encode_line(encoded_str, add_if_not_exist=False)

        tfm_img = self.tfm(image=image)["image"]  # h, w, c
        return {'id': index, 'tfm_img': tfm_img, 'label_ids': input_ids}


    def mergeBackground(self,image):
        ##隔100次重新载入背景图片

        if self.countMergeIndex!=0:
            background = self.backgroundImage 
        else:
            imagePath = np.random.choice(self.background)
            background = Image.open(imagePath)
            self.backgroundImage = background
        self.countMergeIndex = self.countMergeIndex + 1
        self.countMergeIndex = self.countMergeIndex % 100

        foreground = Image.fromarray(image)
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
        if random.random()>0.6 and self.split=="train":
            merged = elastic_transform(np.array(merged), alpha=300, sigma=8)
        
        return np.array(merged)


    def size(self, idx):
        img_dict = self.data[idx]

        encoded_str = img_dict['encoded_str']
        input_ids = self.target_dict.encode_line(encoded_str, add_if_not_exist=False)
        return len(input_ids)

    def num_tokens(self, idx):
        return self.size(idx)

    def collater(self, samples):
        return default_collater(self.target_dict, samples)

if __name__=="__main__":
    from fairseq.data import Dictionary

    from data.data_aug import build_synthesizer_aug,build_formular_aug
    import uuid
    target_dict = Dictionary.load("dictionary/mopai_chinese_support.txt")
    tfm = build_synthesizer_aug(mode="train")
    formular= SynthesizerRealFormular(gt_path="/home/public/yushilin/formular/taojuan/",tfm=tfm,target_dict=target_dict,input_size=(224,672),split="train",backGroundPath="/home/public/yushilin/formular/back_ground/")
    for f in formular:
        #pass
        print(f["tfm_img"].shape)
        cv2.imwrite("result/{}.jpg".format(uuid.uuid1()),f["tfm_img"])
    # image = cv2.imread("/home/public/yushilin/handwrite/HandWriteFormula/train/math/images/000020.png")
    # image = rotate_image(image,-25)
    # cv2.imwrite("nn.jpg",image)

