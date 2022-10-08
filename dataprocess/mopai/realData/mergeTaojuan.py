import imp
from lib2to3.pytree import type_repr
from os import device_encoding
from pathlib import Path
import argparse
import pathlib
import numpy as np
def changelineID(line,taojuanLen):
    file_name,lineID = line.split(" ")
    lineID = str(int(lineID)+taojuanLen)
    return file_name+" "+lineID
def main(args):
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True,exist_ok=True)
    taojuan_path = Path(args.taojuan_path)
    mopai_path = Path(args.mopai_path)
    # 读取套卷数据
    taojuan_train = taojuan_path / "im2latex_train_filter.lst"
    taojuan_val = taojuan_path / "im2latex_validation_filter.lst"
    taojuan_test = taojuan_path / "im2latex_test_filter.lst"
    taojuan_formula = taojuan_path / "im2latex_formulas.norm.lst"
    
    taojuan = taojuan_train.read_text().split("\n")
    if taojuan_val.exists(): 
        taojuan+=taojuan_val.read_text().split("\n")
    if taojuan_test.exists():
        taojuan+=taojuan_test.read_text().split("\n")
        
    taojuan = [line for line in taojuan if line]
    taojuan_formula = taojuan_formula.read_text().split("@&#")


    len_taojuan_formula = len(taojuan_formula)

    #读取魔拍数据
    mopai_match = mopai_path / "humanLabelMatch.lst"
    mopai_formula = mopai_path / "humanLabel.lst"
    if not mopai_match.exists():
        mopai_match = mopai_path / "im2latex_train_filter.lst"
        mopai_formula = mopai_path / "im2latex_formulas.norm.lst"
    mopai_match = mopai_match.read_text().split("\n")
    mopai_formula = mopai_formula.read_text().split("\n")

    if not mopai_formula[-1]:
        mopai_formula = mopai_formula[:-1]
    if not mopai_match[-1]:
        mopai_match = mopai_match[:-1]    
    print("mopai_formula[-1]",mopai_formula[-1])
    print("mopai_match[-1]",mopai_match[-1])
    
    
    
    if args.split_test:
        np.random.shuffle(mopai_match)
        train_test_gap = int(len(mopai_match) * 0.9)
        train,test = mopai_match[:train_test_gap],mopai_match[train_test_gap:]
        train = [changelineID(line,len_taojuan_formula) for line in train]
        test = [changelineID(line,len_taojuan_formula) for line in test]
    else:
        train = [changelineID(line,len_taojuan_formula) for line in mopai_match]

    train = taojuan+train
    
    out_train = output_path / "im2latex_train_filter.lst"
    out_train.write_text("\n".join(train))

    if args.split_test:
        out_test = output_path / "im2latex_validation_filter.lst"
        out_test.write_text("\n".join(test))

    out_formula = output_path / "im2latex_formulas.norm.lst"
    out_formula.write_text("@&#".join(taojuan_formula+mopai_formula))
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mopai_path",type=str,default="/home/public/yushilin/formular/mopai/realData/humanLabeled/readytrain/")
    parser.add_argument("--taojuan_path",type=str,default="/home/public/yushilin/formular/taojuan_chinese_support")
    parser.add_argument("--output_path",type=str,default="/home/public/yushilin/formular/mopai_chinese_support")
    parser.add_argument("--split_test",action="store_true")
    args = parser.parse_args()
    main(args)