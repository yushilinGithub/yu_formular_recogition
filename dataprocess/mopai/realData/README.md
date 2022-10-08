将公式检测的标注中的图片截取出来，其中保存的json文件为：id,image_id,image_name,box信息，图片命名方式为“imageid_id_folder.jpg”,imageid为图片的id，id为框id，folder为train,val,或者test。
```
python getRealData.py
```
用训练好的模型预测截取的图片，并保存成tran.txt,val.txt,test.txt，更换名字用PPOCRLabel（百度ocr标注工具可读取）,
```
python mopaiInference.py
```
将手动更改的数据转化为模型可读数据
```
python paddleOCR2okayocr.py
```
合并套卷的白底黑字数据以及魔拍业务中真实拍的数据
```
python mergeTaojuan.py
```
提取词表
```
python generate_latex_vocab.py --data-path /home/public/yushilin/formular/mopai_chinese_support/im2latex_train_filter.lst --label-path /home/public/yushilin/formular/mopai_chinese_support/im2latex_formulas.norm.lst --output-file ../dictionary/mopai_chinese_support.txt
```
