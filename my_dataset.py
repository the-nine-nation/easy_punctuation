
import random
import string
import re
from zhon.hanzi import punctuation
import json
from transformers import AutoTokenizer
from datas.lib_dict import book_dict as bookdict1
from datas.lib_dict_2 import book_dict as bookdict2
from torch.utils.data import Dataset
import torch
from torch import cat,as_tensor
vle_processor = AutoTokenizer.from_pretrained("./bert-base-chinese")
k_len = 256# 预处理模型
def clip_text(text):  # 将文字变成id
    """
    将编码的id最后接 "eos_token": "[SEP]",由于处理程序自带，因此不需要额外添加。然后分成最多为n的
    :return:一个tuple类型，每个tuple是一个长度至多为max_lens的tensor
    """
    encode_text = vle_processor(text=[text], return_tensors='pt', padding=True)
    encode_ids = encode_text['input_ids']
    att_mask = encode_text['attention_mask']
    return encode_ids, att_mask

def padding_text(id_encode):
    """
    :param id_encode:
    :return: 最终输出结果为inpudid(1,1025)，input_mask(1,1024)
    """
    paddings_num = k_len - id_encode.shape[1]+1
    input_mask=as_tensor([1]*(id_encode.shape[1])).unsqueeze(dim=0)
    padding_0=as_tensor ([0] * paddings_num).unsqueeze(dim=0)
    sample_ids = cat((id_encode,padding_0),dim=-1)
    input_mask = cat((input_mask,padding_0),dim=-1)[:, :-1]
    input_ids=sample_ids[:,:-1]
    target_ids=sample_ids[:,1:]
    return input_ids,input_mask,target_ids
def delete_code(data):
    data_encode =re.compile('[\\x00-\\x08\\x0b-\\x0c\\x0e-\\x1f]').sub('', data)#删除乱码 # 删除乱码

    return data_encode

def remove_punctuation(input_string):
    # 制作一个映射表，所有标点符号都被映射为None
    translator = str.maketrans('', '', string.punctuation)
    # 使用映射表删除所有标点符号
    no_punct = input_string.translate(translator)
    result = re.sub(r'[{}]+'.format(punctuation), '', no_punct)
    return result
def encode2klen(sourcetext,k_len):#将原始矩阵padding至k_len
    paddings_num = k_len - sourcetext.shape[1]+1
    input_mask=as_tensor([1]*(sourcetext.shape[1])).unsqueeze(dim=0)
    padding_0=as_tensor ([0] * paddings_num).unsqueeze(dim=0)
    sample_ids = cat((sourcetext,padding_0),dim=-1)[:,:-1]
    input_mask = cat((input_mask,padding_0),dim=-1)[:, :-1]
    return sample_ids,input_mask


def add_eng(i_tokens):#随机添加英文字母
    assert i_tokens.shape[0] == 1
    rans = random.randint(1, i_tokens.shape[1] - 1)
    r_str = random.randint(1, 21128)
    i_tokens[0,rans]=r_str
    return i_tokens
rand_punc=[",",".","/","?","[","]","!","！","？","【","】","，","。","、","·","~","——","-"]
def add_punctuation(i_tokens):
    if i_tokens.shape[1]>15:
        ran2 = random.randint(1, (i_tokens.shape[1] - 2))
        return torch.cat((i_tokens[:ran2], i_tokens[ran2+1:]))
    else:
        return i_tokens

def mix_pos(i_tokens):
    ran1 = random.sample(range(2, i_tokens.shape[1] - 2), 2)
    aa,bb=i_tokens[0,ran1[0]],i_tokens[0,ran1[1]]
    i_tokens[0,ran1[1]]=aa
    i_tokens[0, ran1[0]] = bb
    return i_tokens
mix_method = [1, 2, 3]
def mix_token(src_token,rating=0.2):
    true_rate=int(src_token.shape[1]*rating)

    if true_rate>2 and random.random()>0.6:#对于长度大于10的，进行混淆
        for i in range(true_rate):
            rand_method=random.choice(mix_method)
            if rand_method==1:
                src_token=add_eng(src_token)
            elif rand_method==2:
                src_token=add_punctuation(src_token)
            elif rand_method==3:
                src_token=mix_pos(src_token)
    return src_token


class Dataset_novel(Dataset):
    def __init__(self,k_len=256):
        book_list = list(bookdict1.keys())
        id_list = []
        mask_list = []
        target_list = []  # 结果矩阵
        liteid=[]
        litemask=[]
        for key in book_list:
            try:
                f=open(key, "r", encoding='utf-8')# 打开文本
                main_text = f.read()
            except:
                f = open(key, "r", encoding='ISO-8859-1')  # 打开文本
                main_text = f.read()
            main_text=delete_code(main_text)
            main_list=main_text.split("\n")
            f.close()
            for one_passage in main_list:
                if one_passage!="":
                    encode_id, att_mask = clip_text(one_passage)
                    srcmask=(encode_id < 28 )| (encode_id > 100)#删除unknow
                    encode_id=encode_id[srcmask].unsqueeze(0)
                    att_mask=torch.ones(encode_id.size())
                    """
                    从这里删除id>168 && id<667的部分作为无符号输入
                    """
                    if encode_id.shape[1] > k_len:
                        id_list = id_list + list(torch.split(encode_id[:, :-1], k_len, dim=1))[:-1]  # 转化成列表并除去最后那个短样本
                        mask_list = mask_list + list(torch.split(att_mask[:, :-1], k_len, dim=1))[:-1]
                        encode_one=encode_id[:, -k_len - 1:-1]
                        attmask_one=att_mask[:, -k_len - 1:-1]
                        id_list.append(encode_one)  # 输入矩阵
                        mask_list.append(attmask_one)  # 输入的padding_mask
                        target_list = target_list + list(torch.split(encode_id[:, 1:], k_len, dim=1))[:-1]
                        target_list.append(encode_id[:, -k_len:])  # 结果矩阵
                    elif encode_id.shape[1]>10:
                        input_ids, input_mask, target_ids = padding_text(encode_id)
                        id_list.append(input_ids)
                        mask_list.append(input_mask)
                        target_list.append(target_ids)
        for encode_one in id_list:
            mask = ((encode_one > 169) & (encode_one < 406)) | ((encode_one > 427) & (encode_one < 534)) | (
                        (encode_one > 7993) & (encode_one < 8028)) | (
                           encode_one == 0)
            mask = ~mask
            literes = encode_one[mask].unsqueeze(0)
            #这里对token添加噪点
            literes=mix_token(literes)
            lite_0, lite_mask = encode2klen(literes, k_len=k_len)
            liteid.append(lite_0)
            litemask.append(lite_mask)
        self.id_list = id_list
        self.mask_list=mask_list
        self.target_list=target_list
        self.src_list=liteid
        self.src_mask=litemask

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        return self.id_list[idx],self.mask_list[idx],self.target_list[idx],self.src_list[idx],self.src_mask[idx]
class Dataset_qinghua(Dataset):
    def __init__(self,k_len=256):
        book_list = list(bookdict2.keys())
        id_list = []
        mask_list = []
        target_list = []  # 结果矩阵
        liteid=[]
        litemask=[]
        for key in book_list:
            try:
                f=open(key, "r", encoding='utf-8')# 打开文本
                main_text = f.read()
            except:
                f = open(key, "r", encoding='ISO-8859-1')  # 打开文本
                main_text = f.read()
            main_text=delete_code(main_text)
            main_list=main_text.split("\n")
            f.close()
            for one_passage in main_list:
                if one_passage!="":
                    encode_id, att_mask = clip_text(one_passage)
                    srcmask=(encode_id < 28 )| (encode_id > 100)#删除unknow
                    encode_id=encode_id[srcmask].unsqueeze(0)
                    att_mask=torch.ones(encode_id.size())
                    """
                    从这里删除id>168 && id<667的部分作为无符号输入
                    """
                    if encode_id.shape[1] > k_len:
                        id_list = id_list + list(torch.split(encode_id[:, :-1], k_len, dim=1))[:-1]  # 转化成列表并除去最后那个短样本
                        mask_list = mask_list + list(torch.split(att_mask[:, :-1], k_len, dim=1))[:-1]
                        encode_one=encode_id[:, -k_len - 1:-1]
                        attmask_one=att_mask[:, -k_len - 1:-1]
                        id_list.append(encode_one)  # 输入矩阵
                        mask_list.append(attmask_one)  # 输入的padding_mask
                        target_list = target_list + list(torch.split(encode_id[:, 1:], k_len, dim=1))[:-1]
                        target_list.append(encode_id[:, -k_len:])  # 结果矩阵
                    elif encode_id.shape[1]>12:
                        input_ids, input_mask, target_ids = padding_text(encode_id)
                        id_list.append(input_ids)
                        mask_list.append(input_mask)
                        target_list.append(target_ids)
        for encode_one in id_list:
            mask = ((encode_one > 169) & (encode_one < 406)) | ((encode_one > 427) & (encode_one < 534)) | (
                        (encode_one > 7993) & (encode_one < 8028)) | (
                           encode_one == 0)
            mask = ~mask
            literes = encode_one[mask].unsqueeze(0)
            #这里对token添加噪点
            literes=mix_token(literes)
            lite_0, lite_mask = encode2klen(literes, k_len=k_len)
            liteid.append(lite_0)
            litemask.append(lite_mask)
        self.id_list = id_list
        self.mask_list=mask_list
        self.target_list=target_list
        self.src_list=liteid
        self.src_mask=litemask

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        return self.id_list[idx],self.mask_list[idx],self.target_list[idx],self.src_list[idx],self.src_mask[idx]

