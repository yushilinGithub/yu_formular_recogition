import argparse
from pathlib import Path
from utils.latex2png import Latex
import cv2
from PIL import Image
import numpy as np
import logging
from tqdm import tqdm
from multiprocessing import Pool
import os
def draw(args,math,path,lineID):
    math = "$$"+math+"$$"
    math = r"{}".format(math)
    pngs = Latex(math,line=lineID).write(return_bytes=False)
  
    try:
        pngs = cv2.imread(str(pngs),cv2.IMREAD_GRAYSCALE)
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
def get_formula(path):
    result = []
    with open(path,"r",encoding="utf-8") as f:
        data = f.readlines()
    for line in data:
        lineInfo = eval(line)
        result.append(lineInfo["Label"])
    return result
def main(args):
    TAL_OCR_path = Path(args.TAL_OCR_path) 
    output_path = Path(args.output_path)/"formula_images_processed"
    if not output_path.is_dir():
        output_path.mkdir()
    train = TAL_OCR_path/"train.txt"
    val = TAL_OCR_path/"val.txt"
    formulaList = get_formula(train)+get_formula(val) # concat train.txt and val.txt together. 
    img_name = "000000000"
    formulas = []
    result_paths = []
    for i,formula in enumerate(formulaList):
        image_name = img_name[:-len(str(i))]+str(i)+".jpg"
        img_path = output_path / image_name
        print(img_path)
        if os.path.exists(img_path):
            #draw(args,formula,str(img_path),i)
            result_path = "TAL/"+image_name
            match = result_path + " " +str(len(formulas))
            formulas.append(formula)

            result_paths.append(match)
    with open(os.path.join(args.output_path,"im2latex_formulas.norm.lst"),"w") as f:
        f.write("@&#".join(formulas))
    with open(os.path.join(args.output_path,"im2latex_train_filter.lst"),"w") as f:
        f.write("\n".join(result_paths))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--TAL_OCR_path",type=str,default="/home/public/yushilin/formular/TAL_OCR_formula")
    parser.add_argument("--output_path",type=str,default="/home/public/yushilin/formular/TAL_formula_1")
    parser.add_argument('-d', '--divable', type=int, default=32, help='To what factor to pad the images')
    args = parser.parse_args()
    main(args)