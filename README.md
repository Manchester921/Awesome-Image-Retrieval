# Awesome-Image-Retrieval


![image-20220107160728854](pic\image-20220107160728854.png)

[TOC]

#  一、概述

> [图像检索: yongyuan博客]( https://yongyuan.name/blog/)
> [深度学习图像搜索与识别](https://blog.csdn.net/Sophia_11/article/details/117408262 )
> [【TPAMI重磅综述】 SIFT与CNN的碰撞：万字长文回顾图像检索任务十年探索历程（上篇）](https://mp.weixin.qq.com/s/sM78DCOK3fuG2JrP2QaSZA) 
> [【TPAMI重磅综述】 SIFT与CNN的碰撞：万字长文回顾图像检索任务十年探索历程（下篇）](https://mp.weixin.qq.com/s/yzVMDEpwbXVS0y-CwWSBEA) 
> [深度学习图像搜索与识别 豆瓣](https://book.douban.com/subject/35430409/ )
> [gitub : awesome-cbir-papers](https://github.com/willard-yuan/awesome-cbir-papers)
> [图像检索论文博客](https://blog.csdn.net/qq_33208851/category_9314984.html)
> [何恺明编年史](https://mp.weixin.qq.com/s?__biz=MzUxNjcxMjQxNg==&mid=2247538670&idx=2&sn=ed28773633b2a658ead88ffe83f6e457&chksm=f9a12561ced6ac77afac01170ae273076fa112199eb7d00cf7a8ef5ac1ebdd2d63fdf8384a51&scene=132#wechat_redirect)




------

# 二、传统图像检索 TBIR  (Text Based Image Retrieval)

![image-20220111094210412](pic\image-20220111094210412.png)

传统图像检索主要的代表方法是SIFT算法。

视觉特征SIFT被证明具有优秀的描述性和区分能力来捕获各种文献中的视觉内容。它能很好地捕捉到旋转和缩放变换的不变性，并对光照变化具有鲁棒性。第二项工作是引入袋式视觉词（BoW）模型。利用信息检索，BoW模型基于包含的局部特征的量化来紧凑地表示图像，并且容易地适应用于可缩放图像检索的经典倒排文件索引结构。


## 2.1 SIFT算法(Scale-invariant feature transform)

<img src="pic\image-20220107161401771.png" alt="image-20220107161401771"  />

特征点检测方法：小波变换、傅里叶变换、高斯差分（DoG），MSER，Hessian仿射检测器，HarrisHessian检测器和FAST 
特点：SIFT特征是图像的局部特征，其对旋转、尺度缩放、亮度变化保持不变性，对视角变化、仿射变换、噪声也保持一定程度的稳定性



> [超强大的SIFT图像匹配技术详细指南（附Python代码) ](https://baijiahao.baidu.com/s?id=1650694563611411654&wfr=spider&for=pc)
> [SIFT算法原理](https://blog.csdn.net/qq_37374643/article/details/88606351 )
> [基于纹理特征的图像检索算法](https://blog.csdn.net/leixiaohua1020/article/details/16859181)   FCTH、CEDD

## 2.2 哈希算法 (hash)

局部敏感哈希算法、ITQ算法、改进PHash 去重 

- 局部敏感哈希算法LSH()
  特点：LSH的一大特点是原始空间相近的两个数据点哈希编码也相似。
  缺点：效率低，且要保证精度需要很长的编码，low recall

- ITQ算法 (Iterative Quantization)
  算法流程：(1)使用PCA、LDA等无监督 或 有监督 提取特征降维后，(2)把原始数据分别映射到超立方体的顶点，即将浮点编码向量学习转化为01二维向量。

> [Iterative Quantization论文理解及代码讲解 ](https://blog.csdn.net/liuheng0111/article/details/52242491 )
> [图像检索哈希算法综述](https://blog.csdn.net/qq_31293215/article/details/89928438)
> [Hashing图像检索源码及数据库总结](https://yongyuan.name/blog/codes-of-hash-for-image-retrieval.html)
> [拷贝检索PHash改进方案](https://yongyuan.name/blog/improve-phash-for-copy-detection.html)



## 2.3 BoW、FV、VLAD算法

- BoF算法 (Bag of visual Feature/word)
  算法原理：方法的核心思想是提取出关键点描述子后利用聚类的方法训练一个码本，随后每幅图片中各描述子向量在码本中各中心向量出现的次数来表示该图片。TF-IDF
  算法流程：(1)利用SIFT算法从不同类别的图像中提取视觉词汇向量，这些向量代表的是图像中局部不变的特征点；(2))将所有特征点向量集合到一块，利用K-Means算法合并词义相近的视觉词汇，构造一个包含K个词汇的单词表；(3)统计单词表中每个单词在图像中出现的次数，从而将图像表示成为一个K维数值向量。
  特点：该方法的缺点是需要码本较大



- FV算法  (Fisher Vector)
  算法原理：FV方法的核心思想是利用高斯混合模型(GMM)，通过计算高斯混合模型中的均值、协方差等参数来表示每张图像。
  特点：该方法的优点是准确度高，但缺点是计算量较大。



- VLAD算法 (vector of locally aggregated descriptors)
  算法原理：用图片特征与各个聚类中心的累加距离向量来表示图像。
  算法流程：(1)读取图片文件路径及特征提取，(2) 使用聚类方法训练码本，(3) 将每张图片的特征与最近的聚类中心进行累加，~~(4)~~对累加后的VLAD进行PCA降维并对其归一化，~~(5)~~得到VLAD后，使用ADC方法继续降低储存空间和提高搜索速度
  特点：相比FV计算量较小，相比BoW码书规模很小，并且检索精度较高。
  <img src="pic\image-20220112103955282.png" alt="image-20220112103955282" style="zoom:67%;" />

> [BOW 原理及代码解析](https://blog.csdn.net/tiandijun/article/details/51143765)
> [Fisher Vector 通俗学习](https://blog.csdn.net/ikerpeng/article/details/41644197 )
> [Fisher Vector基本原理与用法]( https://blog.csdn.net/wzmsltw/article/details/52040010 )
> [Fisher Kernels原理](https://blog.csdn.net/breeze5428/article/details/32706507)
> [BoF、VLAD、FV三剑客](https://yongyuan.name/blog/cbir-bow-vlad-fv.html)
> [图像检索与降维（一）：VLAD](https://blog.csdn.net/LiGuang923/article/details/85416407)






------

# 三、深度学习图像检索 CBIR (Content Based Image Retrieval)



## 3.1 Backbones 

1. VGG

2. ResNet

3. SE-ResNeSt

4. EffectNet

5. ViT

6. DeiT

7. Swin Transformer

   [重磅开源！屠榜各大CV任务！最强骨干网络：Swin Transformer来了](https://mp.weixin.qq.com/s?__biz=MzUxNjcxMjQxNg==&mid=2247520455&idx=3&sn=0813367e77064ebd00afd7fe9f33f509&chksm=f9a1e248ced66b5e2e001270fe018e9ca9f672d69cb3c30d224e3b0062730a130fe2ea524fa0&scene=21#wechat_redirect)

![image-20220111120109401](pic\image-20220111120109401.png)

8. MAE预训练模型
   [Mask的预训练模型 MAE (mask autoencoder)](https://mp.weixin.qq.com/s?__biz=Mzg4MjQ1NzI0NA==&mid=2247497321&idx=1&sn=7df8812e557407dbf889cf0c44432ae3&chksm=cf54d99af823508c8702eb5b2540110a772221ef97733fedf5fb485ca02d4466ed1a7a574ea5&scene=21#wechat_redirect)

![image-20220111114821799](pic\image-20220111114821799.png)

## 3.2 细粒度图像识别检索 FGIA (fine-gained image analysis)

[2019 : Deep learning for fine-grained image analysis: A survey](https://blog.csdn.net/weixin_45691429/article/details/107470493)
[最新的细粒度图像分析资源](https://zhuanlan.zhihu.com/p/73075939)
<img src="pic\image-20220111145258014.png" alt="image-20220111145258014" style="zoom: 80%;" />

- [2019CVPR细粒度论文笔记《Destruction and Construction Learning for Fine-grained Image Recognition》](https://blog.csdn.net/zsx1713366249/article/details/92370490)

主要贡献：提出一种新颖的细粒度图像识别框架，称为“破坏-重建学习” DCL (Destruction and Construction Learning)
动机原理：故意破坏全局结构。对细粒度分类，局部细节相比全局结构起着更重要的作用。
特点：轻量级GAN，容易训练。

![image-20220111152214610](pic\image-20220111152214610.png)

[双线性卷积神经网络模型（Bilinear CNN)](https://blog.csdn.net/weixin_44529634/article/details/106543623)



## 3.3  GAN对抗生成网络

[Binary Generative Adversarial Networks for Image Retrieval](https://blog.csdn.net/qq_33208851/article/details/102542997)   无监督 二进制生成对抗性网络

![image-20220111111940441](pic\image-20220111111940441.png)

## 3.4 多标签学习



## 3.5 图像预处理



## 3.6 loss

- contrastive loss



- margin loss

> [ArcFace算法笔记](https://blog.csdn.net/u014380165/article/details/80645489 )

- triplet ranking loss



- softmax accelerate

> [sampled softmax与其在框架中的使用](https://zhuanlan.zhihu.com/p/129824834)
> [Pytorch-NCE](https://github.com/Stonesjtu/Pytorch-NCE)
> [词嵌入系列博客Part2：比较语言建模中近似softmax的几种方法](https://www.sohu.com/a/117027735_465975?spm=1001.2101.3001.5697)

## 3.7 trick

***[回顾：基于深度学习的图像检索](https://zhuanlan.zhihu.com/p/77429436)**
[基于深度学习的视觉实例搜索研究进展](https://zhuanlan.zhihu.com/p/22265265 )

[layer选择与fine-tuning性能提升验证](https://yongyuan.name/blog/layer-selection-and-finetune-for-cbir.html )

[2017: Fine-tuning CNN Image Retrieval with No Human Annotation](https://www.cnblogs.com/wanghui-garcia/p/13754831.html) 
主要贡献：提出了一种可训练的Generalized-Mean(GeM)池化层，它概括了最大池化和平均池化
$$
\mathrm{f}_{k}^{(g)}=\left(\frac{1}{\left|\mathcal{X}_{k}\right|} \sum_{x \in \mathcal{X}_{k}} x^{p_{k}}\right)^{\frac{1}{p_{k}}}
$$

ensemble

多视图学习利器----CCA

SAM优化器

TTA 

冻结backbond训练

relu改成prelu或者swish等激活函数

加se

加上多尺度信息

## 3.8 展望

- 图神经网 度量学习

  [ICCV 2021 | 复旦&港大提出GraphFPN：用图特征金字塔提升目标检测性能！](https://blog.csdn.net/amusi1994/article/details/119397798)

  本文提出了图特征金字塔网络： GraphFPN，其能够使其拓扑结构适应不同的内在图像结构，并支持跨所有尺度的同步特征交互。

  ![image-20220112130533709](pic\image-20220112130533709.png)



# 四、向量检索

## 4.1 向量检索 

1. 距离
   汉明距离(异或运算)、欧式距离、点积距离、余弦距离

2. 向量簇检索
   BoW(Bag of Words)    TF-IDF 
   [BoW图像检索原理与实战](https://yongyuan.name/blog/CBIR-BoW-for-image-retrieval-and-practice.html )


3. 向量检索召回



KD树、霍夫曼编码
近似最近邻搜索
乘积量化PQ/SQ 
倒排索引KMeans

> [ANN Search ](https://yongyuan.name/blog/ann-search.html)
> 图索引 HNSW
> [OPQ索引与HNSW索引](https://yongyuan.name/blog/opq-and-hnsw.html ) 

## 4.2 Milvus、Faiss

[图片标签及以图搜图场景应用](https://wenjie.blog.csdn.net/article/details/109025115 )
[Faiss入门及应用经验记录](https://zhuanlan.zhihu.com/p/357414033 )
[Guidelines to choose an index](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index )
[The index factory](https://github.com/facebookresearch/faiss/wiki/The-index-factory )
[Milvus 开源向量搜索引擎](https://zhuanlan.zhihu.com/p/90266233 )

## 4.3 召回、排序

reranking




------

# 五、竞赛、数据集


## 5.1 评价标准

- AUC
- recall、Precision、F1 Score、GMeans

- mAP  (Mean Average Precision)平均准确率

$$
mAP =\frac{\sum_{k=1}^{n} P(k) \cdot I(k)}{R}
$$

<img src="pic\image-20220111104213001.png" alt="image-20220111104213001" style="zoom: 50%;" />

- NDCG (Normalized Discounted cumulative gain) 归一化折损累计增益 

$$
N D C G=\frac{1}{N}\left(\sum_{i=1}^{k} \frac{I\left(k\right)}{\log(k+1)}\right)
$$


- QPS 计算效率
- Memory Cost 内存消耗


## 5.2 数据集

> [MNIST手写数字 ](http://yann.lecun.com/exdb/mnist/)
> [CIFAR-10 and CIFAR-100](http://www.cs.toronto.edu/~kriz/cifar.html)
> [Oxford Buildings](https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/) 5K images 
> [ Caltech101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/) 和 [Caltech256](http://www.vision.caltech.edu/Image_Datasets/Caltech256/) 
> [Google Landmarks dataset (GLDv2)](https://github.com/cvdfoundation/google-landmark) 

>  [常用图像库整理](https://yongyuan.name/blog/database-for-cbir.html)

## 5.3 竞赛

- [Google Landmark Retrieval 2021](https://www.kaggle.com/c/landmark-retrieval-2021)

  

- [The 2021 Image Similarity Dataset and Challenge](https://www.drivendata.org/competitions/79/competition-image-similarity-1-dev/)

  [baseline code](https://github.com/facebookresearch/isc2021)

- [淘宝直播商品识别大赛](https://tianchi.aliyun.com/competition/entrance/231772/information)

  该赛题提供了约15W张图片，1W个细粒度的SKU级别的标签，以及360个组别标签，大部分类别的图片数量都少于20张。

  [淘宝直播商品识别: EDA数据研究, Match R-CNN模型](https://tianchi.aliyun.com/forum/postDetail?spm=5176.12586969.1002.3.3bdf780bJO5wS0&postId=94589)

  ![image-20220111173156860](pic\image-20220111173156860.png)

  [淘宝直播商品识别大赛](https://blog.csdn.net/weixin_42926836/article/details/107387737)

  ![image-20220111173237881](pic\image-20220111173237881.png)

- [Products-10K](https://www.kaggle.com/c/products-10k/discussion)

  [冠军方案分享：ICPR 2020大规模商品图像识别挑战赛冠军解读](https://mp.weixin.qq.com/s/ySmlN5_hHVVFn9hB-jrRHw)

  [1st place solution summary](https://www.kaggle.com/c/products-10k/discussion/188026)

  1. 骨架网络：resnest作为基础骨架网络进行特征提取。——特征提取
  2. 归一化：GeM pooling对骨架网络最后一层特征进行池化。——最大值池化与平均池化的权衡。
  3. 分类器增强：分类器之前引入了一个BNNeck的结构。——增加类间距离
  4. 分类器：Cosface和CircleSoftmax作为最后的损失。——使用cos角度相似度度量 代替 内积相似度度量
  5. Loss设计：Focal Loss和CrossEntropy Loss联合训练。——减少不平衡类的问题
  6. Unhelpful tricks (performance in single model):
     Larger Backbone and larger Scale
     EfficientNet
     AutoAug
     BNN-Style Mixup

![image-20220111172332080](pic\image-20220111172332080.png)

![image-20220111171802927](pic\image-20220111171802927.png)




------

# 六、工业界产品

## 6.1 拍立淘

> 2021: [10亿级！淘宝大规模图像检索引擎算法设计概览]( https://blog.csdn.net/moxibingdao/article/details/117094847)
> 2017:[首次披露！拍立淘技术框架及核心算法，日均UV超千万](https://developer.aliyun.com/article/161333)


## 6.2 微信扫一扫

> 2019: [微信「扫一扫识物」 的背后技术揭秘](https://mp.weixin.qq.com/s/fiUUkT7hyJwXmAGQ1kMcqQ)
> 2020: [揭秘微信「扫一扫」识物为什么这么快](https://mp.weixin.qq.com/s/EBCcBWob_iFa51-gOVPYQA)

## 6.3 百度云api