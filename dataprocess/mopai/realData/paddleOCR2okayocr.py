#将标注的数据以及之前线上加上公开数据合并到一块，
#输出文件 
# 1,图片，根据paddleocr的标注文件将图片拆分，并保存到指定目录下的paddleocr的标注文件中图片的上一级目录中。
#  2，humanLabel.lst 公式的latex文本
#  3，humanLabelMatch.lst 文本以及图片的对应关系
import os
import argparse
import cv2
def changelineID(line,taojuanLen):
    file_name,lineID = line.split(" ")
    lineID = str(int(lineID)+taojuanLen)
    return file_name+" "+lineID

def cutImage(args,labeldict):
    imagePath = os.path.join(args.dstPath,"image")
    if not os.path.exists(imagePath):
        os.mkdir(imagePath)
    formularList = []
    matchList = []
    for filename,Imagedict in labeldict.items():
        image =cv2.imread(os.path.join(args.srcImage,filename))
        subId = 0
        for ldict in Imagedict:

            (x1,y1),_,(x2,y2),_ = ldict["points"]
            if x2>x1 and y2>y1:
                subImageName = filename.split(".")[0]+"_"+str(subId)+".jpg"
                dirname = os.path.dirname(subImageName)
                if dirname.isnumeric():
                    subImageName = os.path.join("train2017_"+dirname,os.path.basename(subImageName))
                formularLineID = len(formularList)
                matchList.append(subImageName+" "+str(formularLineID))
                subId=subId+1
                formularList.append(ldict["transcription"])
                
                subImage = image[y1:y2,x1:x2]

                subImagePath = os.path.join(imagePath,subImageName)
                if not os.path.exists(os.path.dirname(subImagePath)):
                    os.mkdir(os.path.dirname(subImagePath))
                if args.saveImage:
                    cv2.imwrite(subImagePath,subImage)
            else:
                print("{}:{}".format(filename,[x1,x2,y1,y2]))
    return formularList,matchList
def main(args):
    if not os.path.exists(args.label):
        raise "{} not exist".format(args.label)
    if not os.path.exists(args.dstPath):
        os.mkdir(args.dstPath)
    labeldict ={}
    with open(args.label, 'r',encoding='utf-8') as f:
        data = f.readlines()
        for each in data:
            file, label = each.split('\t',1)
            if label:
                label = label.replace('false', 'False')
                label = label.replace('true', 'True')
 
                labeldict[file] = eval(label)
                
                #labeldict[file] = json.loads(label)
            else:
                labeldict[file] = []
    formularList,matchList = cutImage(args,labeldict)
    formularFile = os.path.join(args.dstPath,"humanLabel.lst")
    matchFile = os.path.join(args.dstPath,"humanLabelMatch.lst")
    ori_formula = None
    if os.path.exists(formularFile):
        with open(formularFile,"r") as f:
            ori_formula = f.read().split("\n")
        if not ori_formula[-1]:
            ori_formula = ori_formula[:-1]
        formularList = ori_formula+formularList
    if os.path.exists(matchFile):
        with open(matchFile,"r") as f:
            ori_match = f.read().split("\n")
        if not ori_match[-1]:
            ori_match = ori_match[:-1]  
        matchList =ori_match + [changelineID(line,len(ori_formula)) for line in matchList]
        
    with open(formularFile,"w") as f:
        f.write("\n".join(formularList))
    with open(matchFile,"w") as f:
        f.write("\n".join(matchList))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label",default="/home/public/yushilin/formular/mopai/realData/humanLabeled/annotations/paddleOcrLabel_train2017_1.txt",type=str)
    parser.add_argument("--srcImage",default="/home/public/dataset/ques_formula_12345_hx/")
    parser.add_argument("--dstPath",default="/home/public/yushilin/formular/mopai/realData/humanLabeled/readytrain",type=str)
    parser.add_argument("--saveImage",action="store_true",help="whether save subimage")
    args = parser.parse_args()
    main(args)