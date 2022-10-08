import argparse
from pathlib import Path
import numpy
import cv2
def main(args):
    dir = Path(args.data_path)
    out_dir = Path(args.output_path)
    if not out_dir.exists():
        out_dir.mkdir()
    train_file =  dir/"im2latex_formulas.norm.lst"
    train_file_filter = dir/ "im2latex_train_filter.lst"
    val_file_filter = dir/ "im2latex_validation_filter.lst"
    
    train = train_file_filter.read_text().split("\n")
    val = val_file_filter.read_text().split("\n")
    
    train_filtered = []
    val_filtered = []
    image_dir = dir / "formula_images_processed"
    for line in train:
        image_name = line.split(" ")[0]
        image_path = image_dir / image_name
        if image_path.exists() and isinstance(cv2.imread(str(image_path)),numpy.ndarray):
            train_filtered.append(line)
        else:
            print(image_name)
    for line in val:
        image_name = line.split(" ")[0]
        image_path = image_dir / image_name
        if image_path.exists() and isinstance(cv2.imread(str(image_path)),numpy.ndarray):
            val_filtered.append(line)
        else:
            print(image_name)
    out_train = out_dir/"im2latex_train_filter.lst"
    out_val = out_dir/"im2latex_validation_filter.lst"
    
    out_train.write_text("\n".join(train_filtered))
    out_val.write_text("\n".join(val_filtered))
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",type=str,default="/home/public/yushilin/formular/TAL_FORMULAE_MOPAI")
    parser.add_argument("--output_path",type=str,default="/home/public/yushilin/formular/TAL_FORMULAE_MOPAI/filtered")
    args = parser.parse_args()
    main(args)
