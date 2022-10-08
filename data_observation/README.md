# data observation

### 依赖安装包
- 1,https://miktex.org/download
- 2,https://imagemagick.org/index.php
- 3,https://www.ghostscript.com/

### 这个文件夹下面的文件起到的主要作用是对模型预测的数据进行观察，使用方法如下。
 1, 使用模型生成的LaTeX和真实的LaTeX计算Bleu4,对其进行排序，由低到高。
   ```
   python ref_trans_sort.py
   ```
 2, 对模型生成的LaTeX将其转换为图片，
  ```
    python mv_pic.py
  ```

