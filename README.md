#  图像检索 Image Retrieval

![image-20220113152559120](pic\image-20220113152559120.png)

<!-- ![image-20220107160728854](pic\image-20220107160728854.png) -->

[TOC]

#  一、概述
本文梳理了从传统图像检索到深度图像检索方面的资料链接总结，共节选了150多个链接，主要来自于知乎、CSDN、微信、github等。

资料筛选准则：
尽可能找源文章，
尽可能找较新，
总结的较为全面的文章博客。

站在巨人的肩膀上，可以看的更远。

免责声明
如有冒犯到你的版权，请和我联系删除。


## 1.1 文章

[2018：【TPAMI重磅综述】 SIFT与CNN的碰撞：万字长文回顾图像检索任务十年探索历程（上篇）](https://mp.weixin.qq.com/s/sM78DCOK3fuG2JrP2QaSZA) [（下篇）](https://mp.weixin.qq.com/s/yzVMDEpwbXVS0y-CwWSBEA) 

[2021：基于深度学习的基于内容的图像检索技术：十年调研（2011-2020）](https://zhuanlan.zhihu.com/p/338845142)

[2020：关于服装图像检索的文献综述](https://zhuanlan.zhihu.com/p/266865907)

[2021：2021图像检索综述](https://blog.csdn.net/oYeZhou/article/details/117081654)

[2019：基于内容的图像检索技术综述-CNN方法](https://zhuanlan.zhihu.com/p/42237442)

[2019：基于内容的图像检索技术综述 传统经典方法](https://zhuanlan.zhihu.com/p/40714398)

[2018：基于内容的图像检索技术：从特征到检索](https://zhuanlan.zhihu.com/p/46735159)

[2016：基于深度学习的视觉实例搜索研究进展](https://zhuanlan.zhihu.com/p/22265265 )

[2022：何恺明编年史](https://zhuanlan.zhihu.com/p/415353143?ivk_sa=1024320u)

## 1.2 博客

[2022：gitub : awesome-cbir-papers](https://github.com/willard-yuan/awesome-cbir-papers)

[2014-2019：图像检索: yongyuan博客](https://yongyuan.name/blog/)

[2019：图像检索论文博客](https://blog.csdn.net/qq_33208851/category_9314984.html)

[2018-2019：细粒度图像分类](https://www.zhihu.com/column/c_1033661066437419008)

[2020：Fine-Grained Vision](https://www.zhihu.com/column/c_1351291598479777792)

[2018：图像检索TTdreamloong的博客](https://blog.csdn.net/ttdreamloong/category_7560698.html)

## 1.3 图书

[2021：深度学习图像搜索与识别 豆瓣](https://book.douban.com/subject/35430409/ )

## 1.4 视频课程

[2020：深度学习之以图搜图实战（PyTorch + Faiss)](https://edu.csdn.net/course/detail/31077)

[2020：深度学习之多标签图片分类](https://edu.csdn.net/course/detail/30188)

[2020：深度学习之多目标输出图片分类](https://edu.csdn.net/course/detail/30928)

[2021：深度学习图像搜索与识别](https://www.bilibili.com/video/BV1XN411Z7mh)


------

# 二、传统图像检索 TBIR  (Text Based Image Retrieval)

![image-20220111094210412](pic\image-20220111094210412.png)

## 2.1 SIFT算法(Scale-invariant feature transform)

  特征点检测方法：小波变换、傅里叶变换、高斯差分（DoG），MSER，Hessian仿射检测器，HarrisHessian检测器和FAST 
  特点：SIFT特征是图像的局部特征，其对旋转、尺度缩放、亮度变化保持不变性，对视角变化、仿射变换、噪声也保持一定程度的稳定性

  <img src="pic\image-20220107161401771.png" alt="image-20220107161401771"  />

  [2019：超强大的SIFT图像匹配技术详细指南（附Python代码)](https://baijiahao.baidu.com/s?id=1650694563611411654&wfr=spider&for=pc)

  [2019：SIFT算法原理](https://blog.csdn.net/qq_37374643/article/details/88606351 )

  [2013：基于纹理特征的图像检索算法](https://blog.csdn.net/leixiaohua1020/article/details/16859181)   




## 2.2 BoW、FV、VLAD算法
  [2015：BoF、VLAD、FV三剑客](https://yongyuan.name/blog/cbir-bow-vlad-fv.html)

1. BoF算法 (Bag of visual Feature/word)

  算法原理：方法的核心思想是提取出关键点描述子后利用聚类的方法训练一个码本，随后每幅图片中各描述子向量在码本中各中心向量出现的次数来表示该图片。
  特点：该方法的缺点是需要码本较大
  TF-IDF (term frequency - inverse document frequency)：词频-逆向文件频率，字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降。
  
  [2013：Bag-of-words模型入门介绍文章](http://blog.csdn.net/assiduousknight/article/details/16901427)
  
  [2015：BoW图像检索原理与实战](https://yongyuan.name/blog/CBIR-BoW-for-image-retrieval-and-practice.html )
  
  [2016：BOW 原理及代码解析](https://blog.csdn.net/tiandijun/article/details/51143765)

2. FV算法  (Fisher Vector)

  算法原理：FV方法的核心思想是利用高斯混合模型(GMM)，通过计算高斯混合模型中的均值、协方差等参数来表示每张图像。
  特点：该方法的优点是准确度高，但缺点是计算量较大。

  [2014：Fisher Vector 通俗学习](https://blog.csdn.net/ikerpeng/article/details/41644197 )
  
  [2014：Fisher vector coding Fisher Kernels](https://blog.csdn.net/breeze5428/article/details/32706507)
  
  [2016：Fisher Vector基本原理与用法](https://blog.csdn.net/wzmsltw/article/details/52040010 )


3. VLAD算法 (vector of locally aggregated descriptors)

  算法原理：用图片特征与各个聚类中心的累加距离向量来表示图像。
  算法流程：(1)读取图片文件路径及特征提取，(2) 使用聚类方法训练码本，(3) 将每张图片的特征与最近的聚类中心进行累加，(4)对累加后的VLAD进行PCA降维并对其归一化，(5)得到VLAD后，使用ADC方法继续降低储存空间和提高搜索速度
  特点：相比FV计算量较小，相比BoW码书规模很小，并且检索精度较高。

  <img src="pic\image-20220112103955282.png" alt="image-20220112103955282" style="zoom:67%;" />

  [2018：图像检索与降维（一）：VLAD](https://blog.csdn.net/LiGuang923/article/details/85416407)

## 2.3 哈希算法 (hash)

局部敏感哈希算法、ITQ算法、改进PHash 去重 

- 局部敏感哈希算法LSH

  特点：LSH的一大特点是原始空间相近的两个数据点哈希编码也相似。
  缺点：效率低，且要保证精度需要很长的编码，low recall

- ITQ算法 (Iterative Quantization)

  算法流程：(1)使用PCA、LDA等无监督 或 有监督 提取特征降维后，(2)把原始数据分别映射到超立方体的顶点，即将浮点编码向量学习转化为01二维向量。

  [2016：Iterative Quantization论文理解及代码讲解 ](https://blog.csdn.net/liuheng0111/article/details/52242491 )

  [2019：图像检索哈希算法综述](https://blog.csdn.net/qq_31293215/article/details/89928438)

  [2014：Hashing图像检索源码及数据库总结](https://yongyuan.name/blog/codes-of-hash-for-image-retrieval.html)

  [2018：拷贝检索PHash改进方案](https://yongyuan.name/blog/improve-phash-for-copy-detection.html)


------
# 三、深度学习图像检索 CBIR (Content Based Image Retrieval)



## 3.1 Backbones 

[2020：CNN模型-ResNet、MobileNet、DenseNet、ShuffleNet、EfficientNet](https://mp.weixin.qq.com/s/aNTLkjpV5UdJDhvuzJUD4w)

[2020：经典Backbone简述](https://zhuanlan.zhihu.com/p/158812112)

1. 2014：VGG

2. 2015：ResNet

3. 2017：胶囊网络

4. 2018：EffNet

5. 2020：SE-ResNeSt

6. 2021：ViT

7. 2021：Swin Transformer

  [2021：重磅开源！屠榜各大CV任务！最强骨干网络：Swin Transformer来了](https://blog.csdn.net/amusi1994/article/details/115683688)

  ![image-20220111120109401](pic\image-20220111120109401.png)

## 3.2 无监督预训练模型

1. 2017：BGAN (Binary Generative Adversarial Networks)

  主要贡献：利用无监督的方式实现了图片检索。二进制生成对抗性网络
  [2017 : Binary Generative Adversarial Networks for Image Retrieval](https://blog.csdn.net/qq_33208851/article/details/102542997)   

  ![image-20220111111940441](pic\image-20220111111940441.png)

2. 2021：MAE预训练模型

   [2021：Mask的预训练模型 MAE (mask autoencoder)](https://zhuanlan.zhihu.com/p/435874456)

  ![image-20220111114821799](pic\image-20220111114821799.png)

  <img src="pic\image-20220113093958320.png" alt="image-20220113093958320" style="zoom:80%;" />

## 3.3 细粒度图像识别检索 FGIA (fine-gained image analysis)

  [2019：Deep learning for fine-grained image analysis: A survey](https://blog.csdn.net/weixin_45691429/article/details/107470493)

  [2020：最新的细粒度图像分析资源](https://zhuanlan.zhihu.com/p/73075939)

  [Fine-Grained Vision专栏目录](https://zhuanlan.zhihu.com/p/114218632)

  <img src="pic\image-20220111145258014.png" alt="image-20220111145258014" style="zoom: 80%;" />

1. 2017：双线性聚合CNN

  主要贡献：双线性函数是形如f(x,y)=XAy这样的形式。bilinear pooling主要用于特征融合，对于从同一个样本提取出来的特征 x 和特征 y，通过bilinear pooling得到两个特征融合后的向量，进而用来分类。
  [2019：双线性池化（Bilinear Pooling）详解、改进及应用](https://zhuanlan.zhihu.com/p/62532887)
  [2018：双线性汇合(bilinear pooling)在细粒度图像分析及其他领域的进展综述](https://zhuanlan.zhihu.com/p/47415565)

  ![image-20220112150226310](pic\image-20220112150226310.png)

  ![image-20220113100219162](pic\image-20220113100219162.png)

2. 2019：DCL 网络
  
  主要贡献：提出一种新颖的细粒度图像识别框架，称为“破坏-重建学习” DCL (Destruction and Construction Learning)
  动机原理：故意破坏全局结构。对细粒度分类，局部细节相比全局结构起着更重要的作用。
  特点：轻量级GAN，容易训练。

  [2019：2019CVPR细粒度论文笔记《Destruction and Construction Learning for Fine-grained Image Recognition》](https://blog.csdn.net/zsx1713366249/article/details/92370490)
  <img src="pic\image-20220111152214610.png" alt="image-20220111152214610" style="zoom: 67%;" />




## 3.4 loss

注意距离度量方式 与 检索时的距离度量方式 要相同
对于多标签问题，可以用类标签的汉明距离来替换margin，得到动态margin，进行度量学习

### 3.4.1 类内损失 center loss

- 2016：center loss：用来减少类内的差异，不能有效增大类间的差异性。但不一定适合所有场景

  $$
  \mathcal{L}_{center } = \lambda \sum_{i=1}^{m}\left\|\boldsymbol{x}_{i}-\boldsymbol{c}_{y_{i}}\right\|_{2}^{2}
  $$

  [2017：损失函数改进之Center Loss](https://blog.csdn.net/u014380165/article/details/76946339)

  <img src="pic\image-20220112181642644.png" alt="image-20220112181642644" style="zoom:67%;" />

- 2017：Island loss：在关注类别的类内距离的同时，优化类中心之间的距离。同理，可以考虑多标签类中心距离，相似标签类中心稍微近一些。

  $$
  \mathcal{L}_{Island}=\mathcal{L}_{center}+\lambda_{1} \sum_{\mathbf{c}_{j} \in \mathcal{N}} \sum_{\mathbf{c}_{k} \neq \mathbf{c}_{j} }\left(\frac{\mathbf{c}_{k} \cdot \mathbf{c}_{j}}{\left\|\mathbf{c}_{k}\right\|_{2}\left\|\mathbf{c}_{j}\right\|_{2}}+m\right)
  $$
  [2019：Island Loss](https://blog.csdn.net/u013841196/article/details/89920441)

### 3.4.2 类间损失 margin loss

[2018：ArcFace算法笔记](https://blog.csdn.net/u014380165/article/details/80645489)

[2018：人脸识别论文再回顾之四：cosface](https://zhuanlan.zhihu.com/p/45153595)

- softmax loss:

  $$
  L_{softmax}=-\frac{1}{N} \sum_{i=1}^{N}\log \frac{e^{f_{y_{i}}}}{\sum_{j=1}^{C} e_{j}^{f_{j}}}
  $$

- NSL (Normalized Softmax Loss):

  $$
  L_{NSL}=-\frac{1}{N} \sum_{i}\log \frac{e^{s \cos \left(\theta_{y_{i}, i}\right)}}{\sum_{j} e^{s \cos \left(\theta_{j, i}\right)}}
  $$

- A-Softmax loss:
  $$
  L_{A-Softmax}=-\frac{1}{N} \sum_{i}\log \frac{e^{s\cos \left(\theta_{y_{i}, i}-m\right)}}{e^{s\cos \left(\theta_{y_{i}, i}-m\right)}+\sum_{j \neq y_{i}} e^{s \cos \left(\theta_{j, i}\right)}}
  $$

- LMCL (Large Margin Cosine Loss)：
  $$
  L_{LMCL}=-\frac{1}{N} \sum_{i}\log \frac{e^{s\left(\cos \left(\theta_{y_{i}, i}\right)-m\right)}}{e^{s\left(\cos \left(\theta_{y_{i}, i}\right)-m\right)}+\sum_{j \neq y_{i}} e^{s \cos \left(\theta_{j, i}\right)}}
  $$

  ![image-20220112185125062](pic\image-20220112185125062.png)

### 3.4.3 Pair-based loss

  [2019：度量学习中的pair-based loss](https://zhuanlan.zhihu.com/p/72516633)

  [2020：Multi-Similarity Loss使用通用对加权进行深度度量学习-CVPR2019](https://zhuanlan.zhihu.com/p/108421195)

  [2020：深度度量学习－论文简评](https://zhuanlan.zhihu.com/p/141409820)

  [张俊林：对比学习研究进展精要](https://mp.weixin.qq.com/s/xYlCAUIue_z14Or4oyaCCg)

-  Contrastive Loss / Pairwise Ranking Loss：这种损失函数可以有效的处理孪生神经网络中的paired data的关系。

  $$
  L=\frac{1}{2 N} \sum_{n=1}^{N} y d^{2}+(1-y) \max (\operatorname{margin}-d, 0)^{2}
  $$
  <img src="pic\image-20220112183234401.png" alt="image-20220112183234401" style="zoom:50%;" />

- Triplet Ranking Loss：在triplet loss 中有如下的方式:Offline triplet mining、Online triplet mining

  $$
  L=\sum_{i,j,k}\left(D(x_{i}^{a},x_{j}^{p})-D(x_{i}^{a},x_{k}^{n})+m\right)_{+}
  $$
  [2020：Triplet Loss Ranking Loss and Margin Loss](https://zhuanlan.zhihu.com/p/101143469)

- quadruplet loss：不仅要求 $D(x,x^{p})<D(x,x^{n_1})$，还需要$D(x,x^{p})<D(x^{n_1},x^{n_2})$

  $$
  \begin{aligned}
  L_{\text {quadruplet }}=& \sum_{i, j, k}^{N}\left[D(x_{i},x_{j}^{p})-D(x,x_{k}^{n_1})+\alpha_{1}\right]_{+} \\
  &+\sum_{i, j, k, l}^{N}\left[D(x_{i},x_{j}^{p})-D(x_{k}^{n_1},x_{l}^{n_2})+0.5*\alpha_{2}\right]_{+}
  \end{aligned}
  $$
  使用动态margin，计算的是每一个batch中正例图像组和反例图像组各自的平均距离。
  $$
  \begin{aligned}
  \alpha &=w\left(\mu_{n}-\mu_{p}\right) \\
  &=w\left(\frac{1}{N_{n}} \sum_{i, k}^{N} D\left(x_{i} x_{k}^{n}\right)^{2}-\frac{1}{N_{p}} \sum_{i, j}^{N} D\left(x_{i}, x_{j}^{p}\right)^{2}\right)
  \end{aligned}
  $$

  [2019：Beyond triplet loss: a deep quadruplet network for person re-identification泛读记录](https://blog.csdn.net/CsdnWujinming/article/details/90778936)

- 2021：SimCSE loss：这里的距离要标准化。缩小类间距离，并且拉大当前样本和不相关样本的距离，使其uniformity。 

  $$
  L_{\text {SimCSE}}=-\log \frac{\exp \left( D\left(x_{i}, x_{j}^{p}\right) / \tau \right) }{\sum_{j,k}^{N}\left(\exp \left( D\left(x_{i}, x_{j}^{p}\right) / \tau \right) + \exp \left( D\left(x_{i}, x_{k}^{n}\right) / \tau \right) \right)}
  $$

  ![image-20220114180556333](pic\image-20220114180556333.png)

  [2021：SimCSE对比学习: 文本增广是什么牛马，我只需要简单Dropout两下](https://blog.csdn.net/weixin_45839693/article/details/116302914)

  [2021：真正的利器：对比学习SimCSE](https://www.jianshu.com/p/ebe95c24bac0)

- ***2020：circel loss**: 统一了triplet loss和softmax ce loss，正负样本不平衡也可以
  $$
  \begin{aligned}
  \mathcal{L}_{u n i} &=\log \left[1+\sum_{i=1}^{K} \sum_{j=1}^{L} \exp \left(\gamma\left(s_{n}^{j}-s_{p}^{i}+m\right)\right)\right] \\
  &=\log \left[1+\sum_{j=1}^{L} \exp \left(\gamma\left(s_{n}^{j}+m\right)\right) \sum_{i=1}^{K} \exp \left(\gamma\left(-s_{p}^{i}\right)\right)\right]
  \end{aligned}
  $$
  [2020：如何理解与看待在cvpr2020中提出的circle loss？](https://www.zhihu.com/question/382802283)

- ***2020：Smooth AP**: 排序损失，直接优化mAP指标，但不建议单独使用。

  $$
  A P_{q} \approx \frac{1}{\left|\mathcal{S}_{P}\right|} \sum_{i \in \mathcal{S}_{P}} \frac{1+\sum_{j \in \mathcal{S}_{P}} \mathcal{G}\left(D_{i j} ; \tau\right)}{1+\sum_{j \in \mathcal{S}_{P}} \mathcal{G}\left(D_{i j} ; \tau\right)+\sum_{j \in \mathcal{S}_{N}} \mathcal{G}\left(D_{i j} ; \tau\right)}   , 
  \mathcal{G}(x ; \tau)=\frac{1}{1+e^{\frac{-x}{\tau}}} \\
  
  \mathcal{L}_{Smooth  A P}=\frac{1}{m} \sum_{k=1}^{m}\left(1-A P_{k}\right)  。
  $$

  ![image-20220114161843302](pic\image-20220114161843302.png)

  [2020：Smooth AP 图像检索（ECCV2020）](https://zhuanlan.zhihu.com/p/356868571)

### 3.4.4 不平衡损失

- 2014 Hard Negative Mining：相当于给模型定制一个错题集，在每轮训练中不断“记错题”，并把错题集加入到下一轮训练中，直到网络效果不能上升为止。

- 2016 Online Hard Example Mining, OHEM：将所有sample根据当前loss排序，选出loss最大的N个，其余的抛弃。这个方法就只处理了easy sample的问题。

- 2016 Oline Hard Negative Mining, OHNM， 里使用的一个OHEM变种， 在计算loss时， 使用所有的positive anchor, 使用OHEM选择3倍于positive anchor的negative anchor。同时考虑了类间平衡与easy sample。

- Class Balanced Loss。计算loss时，正负样本上的loss分别计算， 然后通过权重来平衡两者。它只考虑了类间平衡。

- 2017 Focal Loss：不会像OHEM那样抛弃一部分样本， 而是和Class Balance一样考虑了每个样本， 不同的是难易样本上的loss权重是根据样本难度计算出来的。
  
  $$
  \mathcal{L}_{Focal}=-\alpha_{\mathrm{t}}\left(1-p_{\mathrm{t}}\right)^{\gamma} \log \left(p_{\mathrm{t}}\right)
  $$

  <img src="pic\image-20220113111053159.png" alt="image-20220113111053159" style="zoom: 50%;" />

- 2019 GHM-C (Gradient Harmonizing Mechanism) ：根据计算得到的梯度密度直方图，对损失进行梯度均衡机制，梯度密度 

  $$
  \begin{aligned}
  L_{G H M-C} &=\frac{1}{N} \sum_{i=1}^{N} \beta_{i} L_{C E}\left(p_{i}, p_{i}^{*}\right) \\
  &=\sum_{i=1}^{N} \frac{L_{C E}\left(p_{i}, p_{i}^{*}\right)}{G D\left(g_{i}\right)}
  \end{aligned}
  $$

  <img src="pic\image-20220113113640887.png" alt="image-20220113113640887" style="zoom:50%;" />

  [2020：Focal Loss与GHM——解决样本不平衡利器](https://zhuanlan.zhihu.com/p/80594704)

  [2019：解决one-stage目标检测正负样本不均衡的另类方法--Gradient Harmonized](https://blog.csdn.net/watermelon1123/article/details/89362220)

  [2017：视觉分类任务中处理不平衡问题的loss比较](https://blog.csdn.net/weixin_35653315/article/details/78327408)


### 3.4.5 softmax accelerate

  [2020：sampled softmax与其在框架中的使用](https://zhuanlan.zhihu.com/p/129824834)

  [2019：github - Pytorch-NCE](https://github.com/Stonesjtu/Pytorch-NCE)

  [2016：词嵌入系列博客Part2：比较语言建模中近似softmax的几种方法](https://www.sohu.com/a/117027735_465975?spm=1001.2101.3001.5697)

## 3.5 trick

### 3.5.1 图像数据增强

- 数据增强
  - 刚性变化：镜像、翻转、旋转、缩放、平移、随机裁剪、随机擦除...
  - 弹性变换：透视、弹性变换、浮雕锐化纹理变换...
  - 色彩变换：直方图均衡、亮度、色调、饱和度、灰度...
  - 噪声变换：椒盐、高斯、动态模糊...
  - 频率变换：高低通滤波、小波变换...
  - mixup：
  
- 困难负样本： OHEM ( Online Hard Example Mining)、XBM (Cross-Batch Memory for Embedding Learning)

  [2020：OHEM 详解](https://blog.csdn.net/m0_45962052/article/details/105068998)

  [2019：CVPR2016 OHEM详细解析](https://zhuanlan.zhihu.com/p/77975552)

  [2020：白嫖的涨点都不要？读MoCo&XBM有感](https://zhuanlan.zhihu.com/p/145449127)

- label smooth 

- pseudo-label

- GAN：生成新图像、增强图像



### 3.5.2 模型结构增改

- 多标签学习：
  - 一阶策略：忽略和其它标签的相关性，比如把多标签分解成多个独立的二分类问题（简单高效）。
  - 二阶策略：考虑标签之间的成对关联，比如为相关标签和不相关标签排序。
  - 高阶策略：考虑多个标签之间的关联，比如对每个标签考虑所有其它标签的影响（效果最优）。

  [2021：多标签学习的新趋势（2021 Survey TPAMI）](https://zhuanlan.zhihu.com/p/266749365)

  [2018：多标签学习综述（A review on multi-label learning algorithms）](https://blog.csdn.net/csdn_47/article/details/83107268)

- [2017：layer选择与fine-tuning性能提升验证](https://yongyuan.name/blog/layer-selection-and-finetune-for-cbir.html )

  <img src="pic\image-20220114105713428.png" alt="image-20220114105713428" style="zoom: 80%;" />

- 多视图学习利器 CCA 、加上多尺度信息

- re-ranking：

- ensemble：Voting、Averaging、Bagging、Boosting、Stacking

  [【机器学习】模型融合方法概述](https://zhuanlan.zhihu.com/p/25836678)

  [Kaggle机器学习之模型融合（stacking）心得](https://zhuanlan.zhihu.com/p/26890738)


### 3.5.3 模型组件替换

- [2019：回顾：基于深度学习的图像检索](https://zhuanlan.zhihu.com/p/77429436)

- [2017：Fine-tuning CNN Image Retrieval with No Human Annotation](https://www.cnblogs.com/wanghui-garcia/p/13754831.html) 
主要贡献：提出了一种可训练的Generalized-Mean(GeM)池化层，它概括了最大池化和平均池化
$$
\mathrm{f}_{k}^{(g)}=\left(\frac{1}{\left|\mathcal{X}_{k}\right|} \sum_{x \in \mathcal{X}_{k}} x^{p_{k}}\right)^{\frac{1}{p_{k}}}
$$

- relu改成prelu或者swish等激活函数

- 加attention：se-block

- dropout ：

  [2021：Multi-Sample Dropout: SimCSE并不是第一个提出多次Dropout](https://blog.csdn.net/weixin_41232882/article/details/120570054)

  [2021：又是Dropout两次！这次它做到了有监督任务的SOTA](https://zhuanlan.zhihu.com/p/386085252)

  [2021：Dropout视角下的MLM和MAE：一些新的启发](https://zhuanlan.zhihu.com/p/443248807)

### 3.5.4 模型训练策略

使用预训练模型，先用冻结backbond，使用ADAM - softmax快速收敛；后用SGD - triple loss

- 优化器：SGD、Momentum、ADAM、SAM、SWA (stochastic weight averaging)。

  [2020：机器学习不得不知道的提升技巧：SWA与pseudo-label](https://cloud.tencent.com/developer/article/1660971)

- 学习率：warmup、Cosine、 ReduceLROnPlateau

- 冻结backbond训练：减少训练显存消耗，加速收敛

- Early Stopping

- 多GPU训练

  

### 3.5.5 模型测试与后处理

- TTA(Test Time Augmentation)：上下左右翻转、镜像
- 模型压缩：蒸馏、剪枝、量化

## 3.6 展望
- listwise learning：列表法排序学习的基本思路是尝试直接优化像 NDCG（Normalized Discounted Cumulative Gain）这样的指标，从而能够学习到最佳排序结果。

  [2020：pairwise、pointwise 、 listwise算法是什么?怎么理解？主要区别是什么？](https://blog.csdn.net/pearl8899/article/details/102920628)

- 图神经网 度量学习
  [2021：万字综述 21年最新最全Graph Learning算法](https://zhuanlan.zhihu.com/p/372271070)
  
  [2021：ICCV 2021 | 复旦&港大提出GraphFPN：用图特征金字塔提升目标检测性能！](https://blog.csdn.net/amusi1994/article/details/119397798)

  本文提出了图特征金字塔网络： GraphFPN，其能够使其拓扑结构适应不同的内在图像结构，并支持跨所有尺度的同步特征交互。

  <img src="pic\image-20220112130533709.png" alt="image-20220112130533709" style="zoom:80%;" />

- [2019：node2vec: Scalable Feature Learning for Networks](https://zhuanlan.zhihu.com/p/46344860)

  图神经网络随机深度游走。深度优先游走 DFS（Depth-first Sampling）和广度优先游走 BFS（Breadth-first Sampling）

  <img src="pic\image-20220112172656169.png" alt="image-20220112172656169" style="zoom:67%;" />

  <img src="pic\image-20220112172727792.png" alt="image-20220112172727792" style="zoom:50%;" />


- OCR：
  [Github：PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.2)

  [Github：open-mmlab/mmocr](https://github.com/open-mmlab/mmocr)

  [Github：Layout-Parser/layout-parser](https://github.com/Layout-Parser/layout-parser)

- image caption：
  [2020：Image Caption方法总结](https://zhuanlan.zhihu.com/p/155919332)  、 [Image Caption方法总结 （二）](https://zhuanlan.zhihu.com/p/153145011)





# 四、向量检索 (Recall、Ranking)

## 4.1 向量检索

### 4.1.1 距离
  汉明距离(异或运算)、编辑距离、欧式距离、马氏距离、点积距离、余弦距离

### 4.1.2 传统检索
- BoW(Bag of Words)：聚类算法对这些矢量数据进行聚类，聚类中的一个簇代表BoW中的一个视觉词
- TF-IDF (term frequency - inverse document frequency)：词频-逆向文件频率，字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降。

[2015：BoW图像检索原理与实战](https://yongyuan.name/blog/CBIR-BoW-for-image-retrieval-and-practice.html )

### 4.1.3 向量检索召回
  [2017：图像检索 再叙ANN (Approximate Nearest Neighbor) Search ](https://yongyuan.name/blog/ann-search.html)
  [向量检索算法综述](https://blog.csdn.net/lijinwen920523/article/details/116358099)

- KNN (K-Nearest Neighbor)、RNN (Radius Nearest Neighbor)
- KD树、Annoy
- 倒排索引  IVF / KMeans
- 乘积量化  PQ / SQ 
- 局部敏感哈希  LSH 
- 图索引 HNSW

  [2018：图像检索 OPQ索引与HNSW索引](https://yongyuan.name/blog/opq-and-hnsw.html )


## 4.2 向量检索引擎
  [2020：图片标签及以图搜图场景应用](https://wenjie.blog.csdn.net/article/details/109025115 )

### 4.2.1 Faiss

  [2021：Faiss入门及应用经验记录](https://zhuanlan.zhihu.com/p/357414033 )

  [2021：Guidelines to choose an index](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index )

  If below 1M vectors: ...,IVFK,...
  If 1M - 10M: "...,IVF65536_HNSW32,..."
  If 10M - 100M: "...,IVF262144_HNSW32,..."
  If 100M - 1B: "...,IVF1048576_HNSW32,..."

[2021：The index factory](https://github.com/facebookresearch/faiss/wiki/The-index-factory )

### 4.2.2 Milvus

[2019：Milvus 开源向量搜索引擎](https://zhuanlan.zhihu.com/p/90266233 )

### 4.2.3  Proxima 

[2021：比 Faiss 更胜一筹？达摩院自主研发的向量检索引擎 Proxima 首次公开！](https://mp.weixin.qq.com/s/yW7UpKJcaSptokPhDkDtGg)



## 4.3 Recall 

- 基于内容的召回：使用item之间的相似性来推荐与用户喜欢的item相似的item。

- 基于协同过滤的召回：协同过滤主要可以分为基于用户的协同过滤、 基于物品的协同过滤、基于模型的协同过滤（如矩阵分解ALS、SVD、SVD++等等）。

- 基于关联规则召回：基于关联规则召回通常有频繁模式挖掘，如Apriori、Fpgrowth等模型

- 基于深度学习模型的召回： 基于深度学习模型的召回也称之为embedding向量召回(每个user和item在一个时刻只用一个embedding向量去表示)的一些经典方法，其主要思想为：将user和item通过DNN映射到同一个低维度向量空间中，然后通过高效的检索方法去做召回。常见的模型有：NCF模型、Youtube DNN召回、 双塔模型召回、MIND模型等等。

- 基于图模型召回：基于图模型召回有二部图挖掘，如simrank；Graph Embedding模型，如DeepWalk、node2vec等模型。

- 基于用户画像的召回：基于用户画像的召回主要根据用户画像如品牌偏好、颜色偏好、价格偏好等偏好信息召回。

- 基于热度召回：热门商品

  [2021：常用推荐算法实现（包括召回和排序）](https://blog.csdn.net/baidu_28610773/article/details/114398265)

## 4.4 re-ranking


1. 基于传统的机器学习模型：基于传统的机器学习模型如LR、SVM等模型。

2. 基于树模型：基于树模型有GBDT、RandomForest、xgboost等。

3. 基于交叉特征模型：基于交叉特征模型有FM、FFM、LR+GBDT等

  ESIM（Enhanced Sequential Inference Model）

  [2019：文本匹配与ESIM模型详解](https://blog.csdn.net/jesseyule/article/details/100579295)

  [2019：短文本匹配的利器-ESIM](https://zhuanlan.zhihu.com/p/47580077)

4. 基于深度学习模型的排序：基于深度学习模型的排序有Wide&Deep、DCN (Deep & Cross Network)、DeepFM等。


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

- NDCG (Normalized Discounted cumulative gain) 归一化折损累计增益 :

  $$
  N D C G=\frac{1}{N}\left(\sum_{i=1}^{k} \frac{I\left(k\right)}{\log(k+1)}\right)
  $$


- QPS 计算效率

- Memory Cost 内存消耗

## 5.2 数据集
  [1998：MNIST手写数字 ](http://yann.lecun.com/exdb/mnist/)
  [2006： Caltech101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/)  [Caltech256](http://www.vision.caltech.edu/Image_Datasets/Caltech256/) 
  [2007：Oxford Buildings](https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/) 5K images 
  [2009：CIFAR-10 and CIFAR-100](http://www.cs.toronto.edu/~kriz/cifar.html)
  [2019：Google Landmarks dataset (GLDv2)](https://github.com/cvdfoundation/google-landmark) 

  [2014：常用图像库整理](https://yongyuan.name/blog/database-for-cbir.html)

## 5.3 竞赛

- [2021：Google Landmark Retrieval 2021](https://www.kaggle.com/c/landmark-retrieval-2021)

  [2021：Transformer杀疯了！神助力！刚拿下Kaggle这项CV赛事冠军！](https://mp.weixin.qq.com/s/7B3hZUpLtTt8NcGt0c-77w)

  DOLG(Orthogonal Local and Global)模型

  ![image-20220113145014735](pic\image-20220113145014735.png)

  [2020：含噪数据的有效训练，谷歌地标图像检索竞赛2020冠军方案解读](https://blog.csdn.net/moxibingdao/article/details/108656568)

  ![image-20220113145851367](pic\image-20220113145851367.png)



- [2020：淘宝直播商品识别大赛](https://tianchi.aliyun.com/competition/entrance/231772/information)

  该赛题提供了约15W张图片，1W个细粒度的SKU级别的标签，以及360个组别标签，大部分类别的图片数量都少于20张。

  [2021：淘宝直播商品识别: EDA数据研究, Match R-CNN模型](https://tianchi.aliyun.com/forum/postDetail?spm=5176.12586969.1002.3.3bdf780bJO5wS0&postId=94589)

  ![image-20220111173156860](pic\image-20220111173156860.png)

  [2020：淘宝直播商品识别大赛](https://blog.csdn.net/weixin_42926836/article/details/107387737)

  <img src="pic\image-20220111173237881.png" alt="image-20220111173237881" style="zoom:67%;" />

- [2020：Products-10K](https://www.kaggle.com/c/products-10k/discussion)

  [2021：冠军方案分享：ICPR 2020大规模商品图像识别挑战赛冠军解读](https://mp.weixin.qq.com/s/ySmlN5_hHVVFn9hB-jrRHw)

  [2021：1st place solution summary](https://www.kaggle.com/c/products-10k/discussion/188026)

  - Validation set : Random sample from the class with more than 20
  - Augmentation : left-right flip； Random Erase; ColorJitter; RandomCrop; Augmix
  - Pooling: GEM pooling
  - Classifier: Cosface, Arcface, CircleSoftmax
  - Loss : Focal loss and CrossEntropy Loss
  - Optimizer : Adam(3e-4, momentum=0.9, decay=1e-5)
  - Backbone: resnest101(bs=192) , resnest200(bs=128) , resnest269(bs=96)
  - Scale: 448, 512(best scale in our exps), 640
  - Unhelpful tricks (performance in single model): (a)Larger Backbone and larger Scale, (b)EfficientNet, (c)AutoAug, (d)BNN-Style, (e)Mixup.

  <img src="pic\image-20220111172332080.png" alt="image-20220111172332080" style="zoom:67%;" />

  ![image-20220111171802927](pic\image-20220111171802927.png)


- [2021：The 2021 Image Similarity Dataset and Challenge](https://www.drivendata.org/competitions/79/competition-image-similarity-1-dev/)

  [2021：baseline code](https://github.com/facebookresearch/isc2021)
  
  


- [2017-2022：AI CITY CHALLENGE](https://www.aicitychallenge.org/)

  [2020：Github: CVPR 2020 AI城市挑战赛4大赛道团队代码官方汇总](https://github.com/NVIDIAAICITYCHALLENGE/2020AICITY_Code_From_Top_Teams)



- [2021：MMVRAC | ICCV 2021 Person Re-Identification ](https://sutdcv.github.io/multi-modal-video-reasoning/#/)

  [2021 ：ICCV person re-identification（行人重识别）论文总结 part1](https://zhuanlan.zhihu.com/p/421480308)   [part2](https://zhuanlan.zhihu.com/p/424698489)

  

- [2020：Huawei DIGIX Image Retrieval](https://developer.huawei.com/consumer/cn/activity/devStarAI/algo/review.html)

  - BackBone： EfficientNet、DenseNet
  - Pool：Generalized Mean Pooling [5]
  - Head ： BNHead [1]
  - Loss ： Triplet Loss + Arcface or Triplet Loss + Amsoftmax
  - 正则化：dropout
  - 其它组件：RAG、Nonlocal、IBN

  ![image-20220115125251837](pic\image-20220115125251837.png)

  [2020 Huawei DIGIX Image Retrieval 亚军方案分享](https://zhuanlan.zhihu.com/p/303371522)


------
# 六、工业界产品

## 6.1 拍立淘
  [2021：10亿级！淘宝大规模图像检索引擎算法设计概览]( https://blog.csdn.net/moxibingdao/article/details/117094847)
 
  [2017：首次披露！拍立淘技术框架及核心算法，日均UV超千万](https://developer.aliyun.com/article/161333)


## 6.2 微信扫一扫
  [2019：微信「扫一扫识物」 的背后技术揭秘](https://mp.weixin.qq.com/s/fiUUkT7hyJwXmAGQ1kMcqQ)

  [2020：揭秘微信「扫一扫」识物为什么这么快](https://mp.weixin.qq.com/s/EBCcBWob_iFa51-gOVPYQA)

## 6.3 图像搜索api
  [百度智能云图像搜索](https://cloud.baidu.com/product/imagesearch)

  [阿里云图像搜索](https://ai.aliyun.com/imagesearch)

  [华为云图像搜索](https://support.huaweicloud.com/imagesearch/index.html)

