#训练数据的单个文件夹太大，将其拆分成每500张一个文件夹的小文件夹
from pathlib import Path
import shutil

def main():
    numImagePerBatch = 500
    labelPath = Path("/home/public/yushilin/formular/mopai/realData/train.txt")
    imagePath = Path("/home/public/dataset/ques_formula_12345_hx/")
    outputPath = Path("/home/public/yushilin/formular/train2017")

    if not outputPath.exists():
        outputPath.mkdir()
    with open(labelPath,"r",encoding="utf-8") as f:
        data = f.readlines()
    splitLabel = []
    fileSate = []
    for i,each in enumerate(data):
        file, label = each.split('\t',1)
        output = outputPath / str(i//numImagePerBatch)
        cacheFilePath = str(i//numImagePerBatch)+"/"+file.split("/")[1]
        if not output.is_dir():
            output.mkdir()
        shutil.copy(imagePath/file,output)
        splitLabel.append(cacheFilePath+"\t"+label)

        if (i+1)%numImagePerBatch==0:
            with open(output/"Cache.cach","w",encoding="utf-8") as f:
                f.write("".join(splitLabel)) 
            splitLabel = []
    with open(output/"Cache.cach","w",encoding="utf-8") as f:
        f.write("".join(splitLabel)) 
if __name__=="__main__":
    main()