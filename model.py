from tkinter import HIDDEN
from typing_extensions import Self
from requests import head
import torch.nn as nn
import torch
import math
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init(self,d_model,eps=1e-12):
        super(LayerNorm,self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    def forward(self,x):
        mean = x.mean(-1,keepdim=True)
        std = x.std(-1,keepdim=True)
        # '-1' means last dimention

        out = (x-mean)/(std + self.eps)
        out = self.gamma*out+self.beta
        return out

class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention
    Query : given sentence that we focused on (decoder)
    Key   : every sentence to check relationship with Query(encoder)
    Value : every sentence same with Key (encoder)
    """
    def __init__(self):
        super(ScaleDotProductAttention,self).__init__()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self,q,k,v,mask = None,e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head,length,d_tensor]
        batch_size,head,length,d_tensor = k.size()

        # 1. dot product Query with key^T to compute similarity
        k_t = k.transpose(2,3)
        score = (q @ k_t) / math.sqrt(d_tensor) # scaled dot product

        # 2. apply masking (opt)
        # if mask is not None
        # score = score.masked_fill(mask==0,-e)

        # 3. pass them softmax to make [0,1] range
        score = self.softmax(score)

        # 4. multiply with value
        v = score @ v
        return v, score

class MultiHeadAttention(nn.Module):
    
    def __init__(self,d_model,n_head):
        super(MultiHeadAttention,self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)
        self.w_concat = nn.Linear(d_model,d_model)

    def forward(self,q,k,v,mask=None):
        # 1. dot product with weight matirces
        q,k,v = self.w_q(q),self.w_k(k),self.w_v(v)

        # 2. split tensor by number of heads
        q,k,v = self.split(q),self.split(k),self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q,k,v, mask = mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. visulize attention map TODO

        return out

    def split(self,tensor):
        """
        split tensor by number of head
        :param tensor:[batch_size,length,d_model]
        :return:[batch_size,head,length,d_tensor]
        """
        batch_size,length,d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size,length,self.n_head,d_tensor).transpose(1,2)
        # it is similar with group convolutin (split by number of heads)

        return tensor

    def concat(self,tensor):
        """
        inverse function of self.split(tensor: torch.Tensor)
        :param tensor:[batch_size,head,lenght,d_tensor]
        :return:[batch_size,length,d_model]
        """
        batch_size,head,length,d_tensor = tensor.size()
        d_model = head * d_tensor
        tensor = tensor.transpose(1,2).contiguous().view(batch_size,length,d_model)
        return tensor

class PositionwiseFeedForward(nn.Module):

    def __init__(self,d_model,hidden,drop_prob=0.1):
        super(PositionwiseFeedForward,self).__init__()
        self.linear1 = nn.Linear(d_model,hidden)
        self.linear2 = nn.Linear(hidden,d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)
    
    def forward(self,x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):

    def __init__(self,d_model,ffn_hidden,n_head,drop_prob):
        super(EncoderLayer,self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model,n_head=n_head)
        self.norm1     = LayerNorm(d_model=d_model)
        self.dropout1  = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model,hidden=ffn_hidden,drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self,x,s_mask):
        # 1. compute self attention
        _x = x
        x = self.attention(q=x,k=x,v=x,mask=s_mask)

        # 2. add and norm
        x = self.norm1(x + _x)
        x = self.dropout1(x)

        # 3. positionwise feed forward Network
        _x = x
        x = self.ffn(x)

        # 4. add and norm
        x = self.norm2(x + _x)
        x = self.dropout2(x)
        return x

class tuEncoder(nn.Module):

    def __init__(self,input_size,d_model,ffn_hidden,n_head,n_layers,drop_prob):
        super().__init__()
        # input [batch,input_size]
        # output [batch,output_size]
        self.layer1 == nn.Linear(in_features=input_size,out_features=d_model)
        self.layers == nn.ModuleList([EncoderLayer(d_model=d_model,
        ffn_hidden=ffn_hidden,
        n_head=n_head,
        drop_prob=drop_prob)for _ in range(n_layers)])
        self.layer2 = nn.Linear(in_features=d_model*68,out_features=1)

    def forward(self,x,s_mask):
        x = self.layer1(x)
        for layer in self.layers:
            x = layer(x,s_mask) # where s_mask is score mask
        # output [batch,seq cout]
        x = self.layer2(x.reshape(-1,68*1024))

        return x
