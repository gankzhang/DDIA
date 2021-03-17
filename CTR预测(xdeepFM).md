[toc]

#### Factorization Machine

- 用于在数据非常稀疏的情况下。与传统的线性模型相比，因子分解机考虑了特征间的交叉，对所有特征变量交互进行了建模。
- 最简单的模型就是将输入的用户和物品的特征进行归纳，比如用户有年龄，性别，地区，职业等特征，物品有类型，价格，颜色，风格等特征，这些用户和物品的特征都在输入中表示为一位$x_i$，作线性回归$f(\textbf{x})=\sum_iw_ix_i$。
- 但是显然只有线性回归表示能力太弱了，需要互动的特征，即$x_ix_j$。
- FM的化简形式计算

  <img src="https://i.loli.net/2021/03/17/9GhMZlTLRqfdJKN.png" alt="image-20210309232541712" style="zoom: 80%;" />
- 但如果特征一多，可能有的特征对根本没有出现过，也就无法训练出权值，这就需要矩阵完全化，将矩阵表示为两个向量的乘积。$W = VV^T$，$V\in R^{k\times n}$。则$W_{ij} = <v_i,v_j>$，$v_i\in R^k$（k<n，本质上是对SVD分解的近似）

#### Field-aware Factorization Machine

- 将相同性质的特征归于一个同一个场，比如一个多类别特征被编码为10个one-hot特征，这些特征就可以放在同一个field内。
- 这里做的就是使每一维特征对其他每一种field都有一个独特的隐向量。则参数数量从$nk$个增加到$nfk$个，$V$变成了$V^{(1)},V^{(2)},...,V^{(f)}$。
- 这里每一个维特征对同一个field中的特征有同样的隐变量，但是同一个field中的特征不共享这些隐变量。后面DeepFM里，同一个field共享了隐变量（共同决定了隐变量）。

#### Deep&Wide

- 



#### DeepFM

- 对每个稀疏特征进行embedding编码，作为通用做法，同一个field的特征被映射到同一个embedding向量，不同field的embedding被组合起来，即一个样本被映射为f个向量，向量长度为$k$
- <img src="https://i.loli.net/2021/03/17/p98cQVDg73zoFe1.jpg" alt="img" style="zoom:67%;" />
- 在FM层中，用$e_{if_i}$代替$v_ix_i$计算，这里$e$的计算在lookup之后还要乘一次$x$，一次项是$\sum e_{if_i}x_i$，二次项是$(\sum e_{if_i})^2-(\sum e_{if_i}^2)$，一次项和二次项一般拼接起来，而且常常长度为$f$的那一维也不需要求和。
- Deep层中就是几个简单的全连接层，最后和FM层输出加起来做logistic回归
- 应该注意，同一个field的不同特征，如果都不为0，他们在FM层计算前还是相对分开的，只是lookup之后的$e$相同，但还需要乘其本身。不同field的特征，直到最后都是分开的，独立进入回归。



#### PNN

- 认为推荐系统中相比于+代表的OR关系，更多的是$\times$代表的AND关系。
- Product 层中每个节点是两两Field的embedding对应的Product的结果
- Product运算有两种定义，inner product函数和outer product函数，分别对应点乘和矩阵乘法。





#### xDeepFM:

- 《Combining Explicit and Implicit Feature Interactions for Recommender Systems》
- 主要创新是引入CIN结构。$x_0$是D维的embedding矩阵上，每一位置上$m$维向量，和第k个field的embedding的对应位置外积（矩阵乘法），形成$Z^{k+1}$。然后$Z^{k+1}$的每一个矩阵乘法得到的矩阵（共D个），被$H_{k+1}$个卷积映射（Feature map）为一个数，组成$x^{k+1}$，再与$x_0$运算得到$k+2$处的$Z$和$x$。
- <img src="https://i.loli.net/2021/03/17/qUP98ovgE57VzR6.png" alt="image-20210310180607470" style="zoom:70%;" />
- ![image-20210310181304625](https://i.loli.net/2021/03/17/3BkSQlJAenI2a1Y.png)
- 这里让每一个field的中间结果，既向下传递到后续模块，又横向传递到其他field处的中间结果运算，实现了有限高阶和参数共享。
- CIN的时间复杂度为$O(mH^2DT)$，



