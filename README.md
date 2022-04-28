# in-between

#### Environments
`pip install -r requirements.txt`

#### Run
1、下载`LAFAN`数据集，并修改`./model/conf`下配置文件中的`data_path`

2、训练
`python main.py train cgan local`

3、测试
`python test.py train cgan local`

#### Pre-trained Model


#### Metrics
L2P: 先对Global Positions 进行正则化，然后再计算预测值与生成值的L2距离；

L2Q: 计算Global Quaternions 的预测值与生成值的L2距离。

可参考 [LaFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/lafan1/benchmarks.py)中的实现。

![L2P&Q](./images/L2PQ.png)
