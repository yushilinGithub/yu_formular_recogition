from cProfile import label
import os
import argparse
import pandas
from utils.latex2png import Latex
import cv2
from PIL import Image
import numpy as np
import logging
from tqdm import tqdm
from multiprocessing import Pool
def draw(args,math,path,lineID):
    math = "$$"+math+"$$"
    math = r"{}".format(math)
    pngs = Latex(math,line=lineID).write(return_bytes=False)
    try:
        pngs = cv2.imread(pngs,cv2.IMREAD_GRAYSCALE)
        data = np.asarray(pngs)
        # print(data.shape)
        gray = 255*(data < 128).astype(np.uint8)  # To invert the text to white
        coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
        a, b, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
        rect = data[b:b+h, a:a+w]
        im = Image.fromarray(rect.astype(np.uint8)).convert('L')
        dims = []
        for x in [w, h]:
            div, mod = divmod(x, args.divable)
            dims.append(args.divable*(div + (1 if mod > 0 else 0)))

        padded = Image.new('L', dims, 255)
        padded.paste(im, im.getbbox())
        padded.save(path)
    except Exception as e:
        logging.warning("math: %s, error %s"%(math,e))
        pass
def process(args,fileline,diff,lineID):
    linelist =  fileline.split("        ")
    if len(linelist) != 4:
        pass
    png_file,subject,predict,label = fileline.split("        ")
    predict = predict.strip()
    label = label.strip()
    predict_name = png_file.split(".")[0]+"_predict.jpg" if not diff else png_file.split(".")[0]+"_diff_predict.jpg"
    path = os.path.join(args.data_dir,subject,predict_name)
    draw(args=args,math=predict,path=path,lineID=lineID)            
    label_name = png_file.split(".")[0]+"_label.jpg" if not diff else png_file.split(".")[0]+"_diff_label.jpg"
    path = os.path.join(args.data_dir,subject,label_name)
    draw(args=args,math=label,path=path,lineID=lineID)
def main(args):
    same_path = os.path.join(args.data_dir,"same_predict.txt")
    diff_path = os.path.join(args.data_dir,"diff_predict.txt")
    with open(diff_path,"r",encoding="utf-8") as f:
        lines = f.readlines()
        
        for i,ro in tqdm(enumerate(lines)):
            process(args,ro,True,i)

    with open(same_path,"r",encoding="utf-8") as f:
        lines = f.readlines()
        for i,ro in tqdm(enumerate(lines)):
            process(args,ro,False,i)


if __name__ == "__main__":
    logging.basicConfig(filename="convert.log",level=logging.INFO)
    logging.info("Started")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",type=str,default="D:\\workspace\\ocr\\formular\\okay\\test")
    parser.add_argument('-d', '--divable', type=int, default=32, help='To what factor to pad the images')
    args = parser.parse_args()
    main(args)
    logging.info("Finished")