# 依赖库dependency  

>>Augmentor  
>>torchsummary  
>>pip install就可以了

# SIM_dataset_generation_and_net_train

## 1.首先你得先自己下载一个高清图片的数据集，例如DIV2K中HR的图片。  

## 2.修改配置文件[configurarion.ini](https://github.com/zenghui19950202/SIM_dataset_generation_and_net_train/blob/master/configuration.ini)。  
>>>配置文件的参数说明：  
>>>**SourceFileDirectory** 你下载的HR图片所在的文件路径，例如 D:\DataSet\DIV2K\DIV2K_HR 。 这个路径下面最好全是图片，最好不要有子文件夹，没有测试过。  
>>>**sample_num** 通过DIV2K图片生成的data pairs的个数。 一般设置为DIV2K图片数量的4倍以下， 例如有500张HR图，sample_num 小于 2000  
>>>**data_ratio:** 训练数据占总数据的比例。 例如 0.8 表示 训练数据 和 验证数据比例 4:1  
>>>**MAX_EVALS**： 尝试的随机超参数的组数。 会用 MAX_EVALS 组超参数分别训练网络，自动寻找最优超参数。  
## 3.弄好配置文件后，先运行[GenerateSpeckleSIMdata.py](https://github.com/zenghui19950202/SIM_dataset_generation_and_net_train/blob/master/GenerateSpeckleSIMdata.py)，生成仿真数据。  

## 4.然后直接再运行[train_SIM_Unet.py](https://github.com/zenghui19950202/SIM_dataset_generation_and_net_train/blob/master/train_SIM_Unet.py)，就可以寻找最优超参数。  

>>>超参数的搜索范围可以在train_SIM_Unet.py里面进行配置。  
>>>![image](https://github.com/zenghui19950202/SIM_dataset_generation_and_net_train/tree/master/images/parameter_grid.PNG)  
  

