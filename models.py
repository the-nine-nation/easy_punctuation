import torch
from transformers import AutoTokenizer
import torch.nn as nn

class Llaja(nn.Module):
    def __init__(self,voc_length=21128,max_len=256,embedding_dim=512):
        super().__init__()
        self.Linear_output = nn.Linear(embedding_dim, voc_length)  # 我们需要知道字典的总长度
        self.lut = nn.Embedding(128100, embedding_dim, padding_idx=0)  # 词向量编码
        self.wpe = nn.Embedding(max_len, embedding_dim)  # 可学习位置编码
        self.trasformers=nn.Transformer(d_model=512,nhead=16,num_encoder_layers=12,num_decoder_layers=12,activation="gelu",dropout=0.0,batch_first=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pos=torch.arange(0, max_len, dtype=torch.long,device=self.device).unsqueeze(0).cuda()
        self.layernorm=nn.LayerNorm(512)
    def forward(self,src,tgt,src_padmask,tgt_keymask):#
        src_pre=self.lut(src)#对src编码
        src_emb = self.wpe(self.pos)#位置编码
        src = self.layernorm(src_pre+ src_emb)
        tgt_pre = self.lut(tgt)
        tgt_emb = self.wpe(self.pos)
        tgt =self.layernorm(tgt_pre + tgt_emb)
        attn_mask = self.trasformers.generate_square_subsequent_mask(256).to(self.device)
        x=self.trasformers(src=src,tgt=tgt,src_key_padding_mask=src_padmask,tgt_key_padding_mask=tgt_keymask,tgt_mask=attn_mask)
        x=self.Linear_output(x)
        return x

