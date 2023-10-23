# easy_punctuation | 针对中文流式语音识别的自监督训练的标点添加/混淆修正模型

由于主要针对中文，英文说明就不写了 ：）

本项目基于transformer和bert-chinese，训练数据集只需要正常有标点的文本，不用对应lable之类，此外并无额外限制。

目前仍处于项目前中期，请直接下载文件使用，接下来会讲解每个文件的作用。我们并不希望您再安装额外的库。

 

### python直接使用

### 使用fastapi

## 项目构成（非常详细）

![2023-10-23-14-18-07-1698041868596.png](assets\2023-10-23-14-18-07-1698041868596.png)



### bert-base-chinese



本项目仅使用bertbasechinese的tokenizer，目前为区分大小写字母，将26个大写英文放入，具体位置见vocab和tokenizer，其中模型需要自己去huggingface下载，放入本文件夹即可。

若需另外添加特殊字符

        方法1：使用python代码添加，此处不赘述，一搜就有；

        当添加词汇数量较少时，请用方法2：先打开**tokenizer**文件，

修改unusedxx 部分，例如将”[unused87]“改为”questions“，然后打开**vocab**文件：



在对应行下修改，注意不需要减一等计算操作，unused后面跟的多少数字就是多少行。

A punctuation addition library based on the Transformer model

### ### datas



数据集文件，其中lib_dict是指向txt文件具体位置的py文件，用于指定需要将哪些txt文件作为样本。

您可以将自己的数据集或我们为您准备的数据集放入该文件夹内，使用dataclear.py处理
