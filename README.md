# PyTorch
学习PyTorch框架的知识总结以及简单案例

一.PyTorch环境的搭建以及安装：

    创建虚拟环境： mkvirtualenv pytorch_neural_network_envs

    安装pytorch: pip install http://download.pytorch.org/whl/cu75/torch-0.1.11.post5-cp35-cp35m-linux_x86_64.whl

    安装torchvision: pip install torchvision

二.框架学习：

    PyTorch-Learning: 知识点

        model_datas: 主要是保存训练模型文件以及训练结果的展示图

        1.torch_numpy.py: 主要是介绍torch与numpy数据类型以及不同的运算对比

        2.torch_variable.py: 主要介绍torch模块里面使用variable

        3.torch_activation.py: 在pytorch中使用常用的4种激活函数

        4.torch_build_nn_quickly.py: 两种创建神经网络的方式

        5.torch_save_reload.py: torch模型的保存与加载

        6.torch_optimizer.py: 常用的4种优化器构建神经网络以及模型效果对比

三.计算padding以及通道数的公式：

    Height--> H  Kernel_size--> K  Padding--> P  Stride--> S

        padding = [(K - S)/2] (表示取整)

        picture_size = (H - K + 2P) /S + 1

        example： 原图片尺寸 H x W = 28*28

            卷积层： S = 1  K = 5

            则： P = 2 ---> picture_size = 28  所以，卷积层负责改变图片的通道数

            池化层： S = 2 K = 2

            则： P = 0 ---> picture_size = 14 所以，池化层负责特征去噪，提取有效特征














