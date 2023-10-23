from models import  *
from transformers import AutoTokenizer
import torch
from torch import cat,as_tensor
from torch.nn import Softmax

def encode2klen(sourcetext,k_len):#将原始矩阵padding至k_len
    paddings_num = k_len - sourcetext.shape[1]+1
    input_mask_0=as_tensor([1]*(sourcetext.shape[1])).unsqueeze(dim=0).cuda()
    padding_0=as_tensor ([0] * paddings_num).unsqueeze(dim=0).cuda()
    sample_ids = cat((sourcetext,padding_0),dim=-1)
    input_mask = cat((input_mask_0,padding_0),dim=-1)[:, :-1]
    input_mask=~(input_mask.bool())
    input_ids=sample_ids[:,:-1]
    return input_ids,input_mask

def preprocess(text):
    # 对文本进行编码
    encoded_input = tokenizer(text)
    return encoded_input
def text_decode(ids):
    text = tokenizer.decode(ids, skip_special_tokens=True)
    # 输出编码结果
    return text

def text2paddingids(source_text):

    source_res = preprocess(source_text)
    source_id, source_mask = torch.tensor(source_res['input_ids']), torch.tensor(source_res['attention_mask'])
    source_id = source_id.unsqueeze(dim=0).cuda()
    padding_id, padding_mask = encode2klen(source_id, k_len)
    return padding_id,padding_mask
softmax=Softmax(dim=-1)
mymodel=torch.load("./model/model_v1_last.pth").cuda()
mymodel.train()
tokenizer = AutoTokenizer.from_pretrained("./bert-base-chinese")
k_len=256

"""
先把没有标点符号的文字转化为0，然后初始化一个起始值，将每次结果作为下一次输入
"""
def text2loop(loop_text):#获得一个不包含cls的结果
    source_res = preprocess(loop_text)
    source_id, source_mask = torch.tensor(source_res['input_ids']).unsqueeze(dim=0), torch.tensor(source_res['attention_mask']).unsqueeze(dim=0)
    source_id=source_id[:,:-1].cuda()
    padding_id, padding_mask = encode2klen(source_id, k_len)
    return padding_id,padding_mask
def gogo(source_text= "先把没有标点符号的文字转化为0然后初始化一个起始值将每次结果作为下一次输入"):
    s_id,s_mask=text2paddingids(source_text)
    print(text_decode(s_id.tolist()[0]))
    t2=""
    tgt_id,tgt_mask=text2loop(t2)
    start_id=torch.tensor([[101]]).cuda()
    start_point=1
    run_sign=True
    while start_point<k_len:
        loop_res=mymodel(src=s_id.cuda(),tgt=tgt_id,src_padmask=s_mask.cuda(),tgt_keymask=tgt_mask)
        _, predicted = torch.max(loop_res, dim=2)
        results=predicted[:,start_point-1:start_point]
        if results.item()==102 or start_point==256:
            return text_decode(start_id.tolist()[0])
            break
        start_id=cat((start_id,results),dim=1)
        print(text_decode(start_id.tolist()[0]))
        tgt_id,tgt_mask=encode2klen(start_id,k_len)
        start_point+=1
        pass