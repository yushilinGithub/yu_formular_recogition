import os
import argparse
import pandas
from utils.latex2png import Latex
import cv2
from PIL import Image
import numpy as np
import logging
def main(args):
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    df = pandas.read_csv(args.csv_path)
    ind2pic = []
    with open(args.pic_latex,"r") as f:
        for ro in f:
            ind2pic.append(ro.split(" ")[0])
    for i,row in df.iterrows():
        if i>=2000:
            break
        line = row['line']
        score = int(float(row['s'])*100)

        math = "$$"+row["D"].strip()+"$$"
        math = r"{}".format(math)
        pngs = Latex(math,line).write(return_bytes=False)
        if pngs==False:
            continue
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
            padded.save(os.path.join(args.save_dir,"{}_{}_syn.jpg".format(score,line)))
        except Exception as e:
            logging.warning("line: %s, error %s"%(line,e))
            pass

        pic = ind2pic[int(line)]
        src_pic = os.path.join(args.src_dir,pic)
        dst_pic = os.path.join(args.save_dir,"{}_{}.jpg".format(score,line))
        image = cv2.imread(src_pic)
        cv2.imwrite(dst_pic,image)

if __name__ == "__main__":
    logging.basicConfig(filename="convert.log",level=logging.INFO)
    logging.info("Started")
    parser = argparse.ArgumentParser()
    # parser.add_argument("--src_dir",type=str,default="/home/public/yushilin/formular/mopai_chinese_support/formula_images_processed/")
    # parser.add_argument("--pic_latex",type=str,default="/home/public/yushilin/formular/mopai_chinese_support/im2latex_validation_filter.lst")
    # parser.add_argument("--save_dir",type=str,default="/home/public/yushilin/formular/mopai_chinese_support/data_obseration/")
    # parser.add_argument("--csv_path",type=str,default="/home/public/yushilin/formular/results/mopai_chinese_support/data_observation.csv")
    parser.add_argument("--src_dir",type=str,default="Y:\\yushilin\\formular\\mopai_chinese_support\\formula_images_processed\\")
    parser.add_argument("--pic_latex",type=str,default="Y:\\yushilin\\formular\\mopai_chinese_support\\im2latex_validation_filter.lst")
    parser.add_argument("--save_dir",type=str,default="Y:\\yushilin\\formular\\mopai_chinese_support\\data_obseration\\")
    parser.add_argument("--csv_path",type=str,default="Y:\\yushilin\\formular\\results\\mopai_chinese_support\\data_observation.csv")
    parser.add_argument('-d', '--divable', type=int, default=32, help='To what factor to pad the images')
    args = parser.parse_args()
    main(args)
    logging.info("Finished")