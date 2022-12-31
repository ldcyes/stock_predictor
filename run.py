from random import shuffle
import model as md
import get_features as gf
from torch import nn
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import pandas as pd
import matplotlib.pyplot as plt

# basic config
stock_codes = ['QQQ','AMD','NVDA','TSM']
key = 'SOXX'
log_interval = 10
pd.set_option('display.width',1000)
pd.set_option('display.max_rows',None)
np.set_printoptions(threshold=np.inf)

epoch_num = 1
train_start_date = '20150617'
train_end_date = '20150617'
test_start_date = '20150617'
test_end_date = '20150617'

train_xy = gf.create_dataset(stock_codes=stock_codes,targe_key=key,start_date=train_start_date,end_date=train_end_date)
train_x,train_y = gf.dataset_reshape(df=train_xy,target_key=key,stock_codes=stock_codes)
train_x  =  torch.where(torch.isnan(train_x),torch.full_like(train_x,0),train_x)

test_xy = gf.create_dataset(stock_codes=stock_codes,targe_key=key,start_date=test_start_date,end_date=test_end_date)
test_x,test_y = gf.dataset_reshape(df=train_xy,target_key=key,stock_codes=stock_codes)
test_x  =  torch.where(torch.isnan(test_x),torch.full_like(test_x,0),test_x)

print(train_x.shape)
print(train_x.shape)

dataset = TensorDataset(train_x,train_y)
train_loader = DataLoader(dataset=dataset,batch_size=32,shuffle=True,num_workers=2)

model = md.tuEncoder(input_size=21,d_model=1024,ffn_hidden=4096,n_head=16,n_layer =12,drop_prob=0.1)
model.load_state_dic(torch.load("tuEncoder128_32.pt"))
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

train_loss_list=[]

for epoch in range(epoch_num):
    if epoch == epoch_num -1:
        out_arr = torch.empty(0)
        label_arr = torch.empty(0)
    for i ,(inputs,lables) in enumerate(train_loader):
        optimizer.zero_grad()
        out = model.forward(x=inputs,s_mask=null)
        loss = criterion(out,lables)
        print("epoch",epoch,"step",i,"batch",i*len(inputs),"MSE:",loss.item())

        loss.backward()
        optimizer.step()

        if(i%log_interval==0):
            train_loss_list.append(loss.item())

        if(epoch==epoch_num -1):
            out_arr = torch.cat((out.veiw(-1),out_arr),0)
            label_arr = torch.cat((lables.view(-1),label_arr),0)

torch.save(model.state_dict(),'tuEncoder.pt')
plt.plot(train_loss_list[20:-1],'b',label='train_loss')
plt.legent(loc='best')
plt.savefig('train_loss.jpg')


plt.clf()
plt.plot(out_arr.view(-1).data.numpy(),'r',label='train_pred')
plt.plot(label_arr.view(-1).data.numpy(),'r',label='train_true')
plt.legent(loc='best')
plt.savefig('final_train.jpg')

pred_test = model(x=test_x,s_mask=null)

plt.clf()
plt.plot(pred_test.view(-1).data.numpy(),'r',label='test_pred')
plt.plot(test_y.view(-1).data.numpy(),'r',label='test_true')
plt.legent(loc='best')
plt.savefig('test.jpg')