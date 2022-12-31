import efinance as ef
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

stock_codes=['QQQ','AMD','NVDA','TMS']
train_key = 'SOXX'
train_start_date = '20200101'
train_end_date = '20200101'
test_start_date = '20200101'
test_end_date = '20200101'

def show_plot(arr,key_name):
    plt.clf()
    plt.plot(arr,lable=key_name)
    plt.legend(loc='best')
    plt.savefig(key_name+".jpg")

# normlization data
def data_norm(data_in,key,max_value,min_value):
    dataset = data_in[key].astype('float')
    scalar = max_value -min_value
    data_in[str(key)+'_norm'] = list(map(lambda x: x/ scalar,dataset))
    return data_in

def get_bolls(dw):
    dw.loc[:,'upper'],dw.loc[:,'upper'],dw.loc[:,'upper'] = ta.BBANDS(
        dw['收盘'],
        timeperiod=20,
        nbdevup=2,
        nbdevdn=2,
        matype=0
    )
    return dw

# 长上影线
def is_long_upper_shadow(dw):
    dw.loc[:,'is_long_upper_shadow']=(np.array(dw['最高'])/np.array(dw['收盘'])>=1.05)
    return dw

def is_long_lower_shadow(dw):
    dw.loc[:,'is_long_lower_shadow']=(np.array(dw['收盘'])/np.array(dw['最低'])>=1.05)
    return dw 

# boll上轨道 
def is_on_upboll(dw):
    dw.loc[:,'is_on_upboll'] = (np.array(dw['收盘'])>=np.array(dw['upper']))
    return dw

def is_on_downboll(dw):
    dw.loc[:,'is_on_downboll'] = (np.array(dw['收盘'])>=np.array(dw['lower']))
    return dw   

# 放量
def is_double_vol(dw):
    sub_arr0 = np.array(dw['成交额'][1:])
    sub_arr1 = np.array(dw['成交额'].iloc[-1])
    arr0 = np.concatenate([sub_arr0,sub_arr1])
    arr1 = np.array(dw['成交额'])
    dw.loc[:,'is_double_vol'] = arr1 > 2*arr0
    return dw

# 七连阳
def is_7rise(dw):
    dw.loc[:,'is_7rise'] = np.zeros(len(dw))
    for i in range(len(dw)):
        if(i<6):
            dw.loc[i,'is_7rise'] = 0
        else:
            dw.loc[i,'is_7rise'] = (dw['收盘'][i]>dw['开盘'][i]) and (dw['收盘'][i-1]>dw['开盘'][i-1]) and (dw['收盘'][i-2]>dw['开盘'][i-2]) and (dw['收盘'][i-3]>dw['开盘'][i-3]) and (dw['收盘'][i-4]>dw['开盘'][i-4]) and (dw['收盘'][i-5]>dw['开盘'][i-5]) and (dw['收盘'][i-6]>dw['开盘'][i-6])
    return dw

# 七连阳
def is_7down(dw):
    dw.loc[:,'is_7down'] = np.zeros(len(dw))
    for i in range(len(dw)):
        if(i<6):
            dw.loc[i,'is_7down'] = 0
        else:
            dw.loc[i,'is_7down'] = (dw['收盘'][i]<dw['开盘'][i]) and (dw['收盘'][i-1]<dw['开盘'][i-1]) and (dw['收盘'][i-2]<dw['开盘'][i-2]) and (dw['收盘'][i-3]<dw['开盘'][i-3]) and (dw['收盘'][i-4]<dw['开盘'][i-4]) and (dw['收盘'][i-5]<dw['开盘'][i-5]) and (dw['收盘'][i-6]<dw['开盘'][i-6])
    return dw

# 上涨趋势中的第7根阴线
def is_up_7down(dw):
    dw.loc[:,'is_up_7down'] = np.zeros(len(dw))
    for i in range(len(dw)):
        k = 0
        is_up_7down_flag = 0
        for j in range(50):
        # find 1st down
            if(i-j>=0):
                if(dw.loc[i-j,'涨跌幅']<0.0):
                    k=k+1
                if(k>6):
                    if(dw.loc[i,'收盘']/dw.loc[i-j,'收盘']>= 1.2):
                        is_up_7down_flag = 1
                        break
        dw.loc[i,'is_up_7down'] = is_up_7down_flag
    return dw

def is_down_7up(dw):
    dw.loc[:,'is_down_7up'] = np.zeros(len(dw))
    for i in range(len(dw)):
        k = 0
        is_down_7up_flag = 0
        for j in range(50):
        # find 1st down
            if(i-j>=0):
                if(dw.loc[i-j,'涨跌幅']<0.0):
                    k=k+1
                if(k>6):
                    if(dw.loc[i,'收盘']/dw.loc[i-j,'收盘']>= 0.8):
                        is_down_7up_flag = 1
                        break
        dw.loc[i,'is_down_7up'] = is_down_7up_flag
    return dw

def create_dataset(stock_codes,targe_key,start_date,end_date):
    df = {}

    for key in stock_codes+[targe_key]:
        df[key,'day'] = ef.stock.get_quote_history(stock_codes=key,beg=start_date,end=end_date,fqt=1,klt=101)# day
        df[key,'week'] = ef.stock.get_quote_history(stock_codes=key,beg=start_date,end=end_date,fqt=1,klt=102)# week
        df[key,'month'] = ef.stock.get_quote_history(stock_codes=key,beg=start_date,end=end_date,fqt=1,klt=103)# month

        df[key,'day']   = get_bolls(df[key,'day'])
        df[key,'week']  = get_bolls(df[key,'week'])
        df[key,'month'] = get_bolls(df[key,'month'])

        df[key,'day'] = is_long_upper_shadow(df[key,'day'])
        df[key,'day'] = is_long_lower_shadow(df[key,'day'])
        df[key,'day'] = is_on_upboll(df[key,'day'])
        df[key,'day'] = is_on_downboll(df[key,'day'])
        df[key,'day'] = is_double_vol(df[key,'day'])
        df[key,'day'] = is_7rise(df[key,'day'])
        df[key,'day'] = is_7down(df[key,'day'])
        df[key,'day'] = is_down_7up(df[key,'day'])
        df[key,'day'] = is_up_7down(df[key,'day'])
        
        df[key,'week'] = is_long_upper_shadow(df[key,'week'])
        df[key,'week'] = is_long_lower_shadow(df[key,'week'])
        df[key,'week'] = is_on_upboll(df[key,'week'])
        df[key,'week'] = is_on_downboll(df[key,'week'])
        df[key,'week'] = is_double_vol(df[key,'week'])
        df[key,'week'] = is_7rise(df[key,'week'])
        df[key,'week'] = is_7down(df[key,'week'])
        df[key,'week'] = is_down_7up(df[key,'week'])
        df[key,'week'] = is_up_7down(df[key,'week'])

        df[key,'month'] = is_long_upper_shadow(df[key,'month'])
        df[key,'month'] = is_long_lower_shadow(df[key,'month'])
        df[key,'month'] = is_on_upboll(df[key,'month'])
        df[key,'month'] = is_on_downboll(df[key,'month'])
        df[key,'month'] = is_double_vol(df[key,'month'])
        df[key,'month'] = is_7rise(df[key,'month'])
        df[key,'month'] = is_7down(df[key,'month'])
        df[key,'month'] = is_down_7up(df[key,'month'])
        df[key,'month'] = is_up_7down(df[key,'month'])

        feature_list = ['开盘','收盘','最高','最低','成交量','upper','midle','lower']
        for feature_name in feature_list:
            df[key,'day'] = data_norm(df[key,'day'],feature_name)
            df[key,'week'] = data_norm(df[key,'week'],feature_name)
            df[key,'month'] = data_norm(df[key,'month'],feature_name)

    day_max_value,day_min_value,week_max_value,week_min_value,month_max_value,month_min_value = 0,0,0,0,0,0

    for key in stock_codes:
        day_max_value = max(np.max(df[key,'day']['成交额']),day_max_value)
        day_min_value = max(np.min(df[key,'day']['成交额']),day_min_value)
        week_max_value = max(np.max(df[key,'week']['成交额']),week_max_value)
        week_min_value = max(np.min(df[key,'week']['成交额']),week_min_value)
        month_max_value = max(np.max(df[key,'month']['成交额']),month_max_value)
        month_min_value = max(np.min(df[key,'month']['成交额']),month_min_value)
        
def dataset_reshape(df,target_key,stock_codes):

    dataX,dataY = [],[]
    week_index = 0
    month_index = 0

    for i in range(df[stock_codes[0],'day'].shape[0]):
        if(df[stock_codes[0],'day']['日期'][i]>df[stock_codes[0],'week']['日期'][week_index]):
            if((week_index+1)<len(df[stock_codes[0],'week']['日期'])):
                week_index +=1
        if(df[stock_codes[0],'day']['日期'][i]>df[stock_codes[0],'month']['日期'][month_index]):
            if((month_index+1)<len(df[stock_codes[0],'month']['日期'])):
                month_index +=1
        dataX_tmp = []
        for key in stock_codes:
            for day_range in range(7):
                if(i-day_range>=0):
                    dataX_tmp = dataX_tmp +[df[key,'day']['振幅'][i-day_range],
                    df[key,'day']['涨跌幅'][i-day_range],
                    df[key,'day']['换手率'][i-day_range],
                    df[key,'day']['is_long_upper_shadow'][i-day_range],
                    df[key,'day']['is_long_lower_shadow'][i-day_range],
                    df[key,'day']['is_on_upboll'][i-day_range],
                    df[key,'day']['is_on_downboll'][i-day_range],
                    df[key,'day']['is_double_vol'][i-day_range],
                    df[key,'day']['is_7rise'][i-day_range],
                    df[key,'day']['is_7down'][i-day_range],
                    df[key,'day']['is_up_7down'][i-day_range],
                    df[key,'day']['is_down_7up'][i-day_range],
                    df[key,'day']['开盘_norm'][i-day_range],
                    df[key,'day']['收盘_norm'][i-day_range],
                    df[key,'day']['最高_norm'][i-day_range],
                    df[key,'day']['最低_norm'][i-day_range],
                    df[key,'day']['成交量_norm'][i-day_range],
                    df[key,'day']['upper_norm'][i-day_range],
                    df[key,'day']['middle_norm'][i-day_range],
                    df[key,'day']['lower_norm'][i-day_range],
                    df[key,'day']['成交额_norm'][i-day_range]]
                else:
                    dataX_tmp = dataX_tmp +21*[0]
            for week_range in range(5):
                if(week_index-week_range>=0):
                    dataX_tmp = dataX_tmp +[df[key,'week']['振幅'][week_index-week_range],
                    df[key,'week']['涨跌幅'][week_index-week_range],
                    df[key,'week']['换手率'][week_index-week_range],
                    df[key,'week']['week_indexs_long_upper_shadow'][week_index-week_range],
                    df[key,'week']['is_long_lower_shadow'][week_index-week_range],
                    df[key,'week']['is_on_upboll'][week_index-week_range],
                    df[key,'week']['is_on_downboll'][week_index-week_range],
                    df[key,'week']['is_double_vol'][week_index-week_range],
                    df[key,'week']['is_7rise'][week_index-week_range],
                    df[key,'week']['is_7down'][week_index-week_range],
                    df[key,'week']['is_up_7down'][week_index-week_range],
                    df[key,'week']['is_down_7up'][week_index-week_range],
                    df[key,'week']['开盘_norm'][week_index-week_range],
                    df[key,'week']['收盘_norm'][week_index-week_range],
                    df[key,'week']['最高_norm'][week_index-week_range],
                    df[key,'week']['最低_norm'][week_index-week_range],
                    df[key,'week']['成交量_norm'][week_index-week_range],
                    df[key,'week']['upper_norm'][week_index-week_range],
                    df[key,'week']['middle_norm'][week_index-week_range],
                    df[key,'week']['lower_norm'][week_index-week_range],
                    df[key,'week']['成交额_norm'][week_index-week_range]]
                else:
                    dataX_tmp = dataX_tmp +21*[0]
            for month_range in range(5):
                if(month_index-month_range>=0):
                    dataX_tmp = dataX_tmp +[df[key,'month']['振幅'][month_index-month_range],
                    df[key,'month']['涨跌幅'][month_index-month_range],
                    df[key,'month']['换手率'][month_index-month_range],
                    df[key,'month']['month_indexs_long_upper_shadow'][month_index-month_range],
                    df[key,'month']['is_long_lower_shadow'][month_index-month_range],
                    df[key,'month']['is_on_upboll'][month_index-month_range],
                    df[key,'month']['is_on_downboll'][month_index-month_range],
                    df[key,'month']['is_double_vol'][month_index-month_range],
                    df[key,'month']['is_7rise'][month_index-month_range],
                    df[key,'month']['is_7down'][month_index-month_range],
                    df[key,'month']['is_up_7down'][month_index-month_range],
                    df[key,'month']['is_down_7up'][month_index-month_range],
                    df[key,'month']['开盘_norm'][month_index-month_range],
                    df[key,'month']['收盘_norm'][month_index-month_range],
                    df[key,'month']['最高_norm'][month_index-month_range],
                    df[key,'month']['最低_norm'][month_index-month_range],
                    df[key,'month']['成交量_norm'][month_index-month_range],
                    df[key,'month']['upper_norm'][month_index-month_range],
                    df[key,'month']['middle_norm'][month_index-month_range],
                    df[key,'month']['lower_norm'][month_index-month_range],
                    df[key,'month']['成交额_norm'][month_index-month_range]]
                else:
                    dataX_tmp = dataX_tmp +21*[0]
        dataX.append(dataX_tmp)

        if((i+20)<df[target_key,'day'].shape[0]):
            dataY.append(df[target_key,'day']['收盘_norm'][i+20]/df[target_key,'day']['收盘_norm'][i])
        else:
            dataY.append(0)

        dataX=np.array(dataX[102:-20][:])
        dataY=np.array(dataX[102:-20])

        dataX=dataX.astype(float)
        dataY=dataY.astype(float)

        print("batch,cin")
        print(dataX.shape)
        print(dataY.shape)

        dataX = dataX.reshape(-1,int(dataX.shape[1]/21),21)
        dataY = dataY.reshape(-1,1)

        print("batch,seq,cin")
        print(dataX.shape)
        print(dataY.shape)

        data_x = torch.from_numpy(dataX).float()
        data_y = torch.from_numpy(dataY).float()

        return data_x, data_y

debug = 0

if(debug):
    train_xy = create_dataset(stock_codes=stock_codes,start_date=train_start_date,end_date=train_end_date)
    train_x,train_y = dataset_reshape(df=train_xy,target_key=train_key,stock_codes=stock_codes)
    key = 'SOXX'
    period = 'day'

    show_plot(arr=train_xy[key,period]['振幅'],key_name='振幅')
    show_plot(arr=train_xy[key,period]['涨跌幅'],key_name='涨跌幅')
    show_plot(arr=train_xy[key,period]['换手率'],key_name='换手率')
    show_plot(arr=train_xy[key,period]['开盘_norm'],key_name='开盘_norm')
    show_plot(arr=train_xy[key,period]['收盘_norm'],key_name='收盘_norm')
    show_plot(arr=train_xy[key,period]['最高_norm'],key_name='最高_norm')
    show_plot(arr=train_xy[key,period]['最低_norm'],key_name='最低_norm')
    show_plot(arr=train_xy[key,period]['成交量_norm'],key_name='成交量_norm')
    show_plot(arr=train_xy[key,period]['成交额_norm'],key_name='成交额_norm')
    show_plot(arr=train_xy[key,period][''],key_name='')
    show_plot(arr=train_xy[key,period]['is_long_upper_shadow'],key_name='is_long_upper_shadow')
    show_plot(arr=train_xy[key,period]['is_lone_lower_shadow'],key_name='is_lone_lower_shadow')
    show_plot(arr=train_xy[key,period]['is_on_upboll'],key_name='is_on_upboll')
    show_plot(arr=train_xy[key,period]['is_on_downboll'],key_name='is_on_downboll')
    show_plot(arr=train_xy[key,period]['is_double_vol'],key_name='is_double_vol')
    show_plot(arr=train_xy[key,period]['is_7rise'],key_name='is_7rise')
    show_plot(arr=train_xy[key,period]['is_7down'],key_name='is_7down')
    show_plot(arr=train_xy[key,period]['is_up_7down'],key_name='is_up_7down')
    show_plot(arr=train_xy[key,period]['is_down_7up'],key_name='is_down_7up')





        