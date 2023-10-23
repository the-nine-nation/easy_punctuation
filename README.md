# easy_punctuation | 针对中文流式语音识别的自监督训练的标点添加/混淆修正模型

A punctuation addition library based on the Transformer model

由于主要针对中文，英文说明就不写了 ：）

本项目基于transformer和bert-chinese，训练数据集只需要正常有标点的文本，不用对应lable之类，此外并无额外限制。

目前仍处于项目前中期，请直接下载文件使用，接下来会讲解每个文件的作用。我们并不希望您再安装额外的库。

 

## 效果（输入均为无标点）

**短句子**

![7d6ed99935765f45cb615c47137bbf6.png](assets/5873dcb540f89a670a3c59ace16933f6bef76623.png)

**长句子**

![b05cd8a1627a046fef63e2a78ef7813.png](assets/8b3e040b8e9e3eb9447460529b9e105be7d1427d.png)

**中英文夹杂**

![](assets/2023-10-23-16-13-52-1698048736423.png)

**带数字效果也很好**

![3db9d9b056bae919609abdbd3052482.png](assets/36304c8ea32b74b2c68d7fde67832ded26acfef0.png)

**感叹号也可以识别出来**

![](assets/2023-10-23-16-39-20-1698048701212.png)

### python直接使用

下载本项目、下载模型文件（等待上传），将inference文件和对应模型文件拖入您的项目，调用推理方法即可

### 使用fastapi

服务端：扒下来文档放入项目中，直接运行fastapirun.py

客户端：

```python
import asyncio
import aiohttp

async def add_punc(text):
    url = f"http://127.0.0.1:8000/punc/{text}"
    async with aiohttp.ClientSession() as session:
        response = await session.get(url)
        res = await response.text()
        return res
```

修改为自己定的IP和端口即可

### 模型结构



## 项目构成（非常详细）

![2023-10-23-14-18-07-1698041868596.png](assets/0294713d522f56d70ad75fcce63e4a5dc5d7a925.png)

### bert-base-chinese

![2023-10-23-14-20-27-1698042020032.png](assets/166fa0c7f32fe235ea8bbadf0a4c8ae5324a3fe4.png)

本项目仅使用bertbasechinese的tokenizer，目前为区分大小写字母，将26个大写英文放入，具体位置见vocab和tokenizer，其中模型需要自己去huggingface下载，放入本文件夹即可。

若需另外添加特殊字符

        方法1：使用python代码添加，此处不赘述，一搜就有；

        当添加词汇数量较少时，请用方法2：先打开**tokenizer**文件，

![2023-10-23-14-24-46-1698042278379.png](assets/8573bb4c3247105b9db7d8e75cb954b379282ac9.png)

修改unusedxx 部分，例如将”[unused87]“改为”questions“，然后打开**vocab**文件：

![2023-10-23-14-29-38-1698042573284.png](assets/a8670ac7357568e3d1b8b595c0fd109607778c14.png)

在对应行下修改，注意不需要减一等计算操作，unused后面跟的多少数字就是多少行。

### ### datas

![](assets/01b593b92656605f2657dc187761b3bc11332a8f.png)

数据集文件，其中lib_dict是指向txt文件具体位置的py文件，用于指定需要将哪些txt文件作为样本。

您可以将自己的数据集或我们为您准备的数据集放入该文件夹内，使用dataclear.py处理

### inference.py
