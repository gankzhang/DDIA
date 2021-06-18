[toc]

### 知识图谱特征学习

知识图谱在推荐系统中的应用有三类：

依次训练，联合训练，交替训练

<img src="https://upload-images.jianshu.io/upload_images/4155986-2b95f73d788a2357.png?imageMogr2/auto-orient/strip|imageView2/2/format/webp" alt="img" style="zoom: 80%;" />

#### 基础概念

- 知识图谱特征学习为知识图谱的实体和关系学习得到一个低维向量，保持图中原有的结构和语义
- 正如词向量的评价和句向量，这种特征学习的评价是依赖于多个下游任务的
- 基于距离的翻译模型
  - 这类模型将尾结点视为头节点和关系翻译得到的
  - 有些要求头尾节点之差等于关系向量，有些要求这个差线性变换后等于关系向量

- 基于语义的匹配模型
  - 这类模型使用基于相似度的评分函数评估三元组的概率，将实体和关系映射到语义空间中进行相似度量度
  - 训练时构造二分类模型，三元组在KG中则为正类
  - 这类方法的代表有SME、NTN、MLP、NAM等

#### DKN（在新闻领域根据序列和预训练实体向量做CTR）

- 依次训练的模型，**Deep Knowledge-aware Network**
- 这里将entity向量与词向量类比使用，一起用CNN提取特征，并用entity向量的邻域平均构成上下文向量（context embedding）
- **推荐系统在新闻领域**的突出问题
  - 高度时间敏感性，这使基于ID的协同过滤算法失效
  - 用户阅读带有倾向性，预测对候选文章的兴趣是关键
  - 新闻的自然语言高度浓缩，包含大量知识实体，不应该仅仅衡量语义
- 给定用户和他的点击历史，一组标题，每个标题是一个词序列，要预测用户是否会点击一个特定的新闻
- 模型中对句子特征的提取使用CNN，即词向量的拼接上的滑动窗口，再用max-pooling
  - 但具体需要包含词向量中的实体信息，且需要包含对应关系（不能横向拼接）
  - 在实体向量上加一个转换函数$g$，并在新一个维度上拼接
- 用一个Attention层将候选标题在浏览历史中得到用户的兴趣向量，并和候选标题的embedding拼接预测
- <img src="https://i.loli.net/2021/04/01/8sVT6gJIEKj32pC.png" alt="image-20210331172904430"  />

#### Ripplenet（利用知识图谱图结构中用户的点击历史预测CTR）

- 这是一个联合学习知识图谱表示和推荐系统的模型
- 一般的特征学习得到的entity和relation的向量是为一般下游任务准备的，但如果专为推荐任务学习一个特征表示，或许有改进
- 各种entity有些是物品，有些是物品的属性，它们构成一个图。可以想象用户的兴趣以其历史记录为中心在知识图谱上逐层向外扩散并衰减，就像Ripple一样
- RippleNet以一个物品和一个用户作为输入，输出用户点击的概率
- 基本思路是对于新入的item，以用户已有的点击历史为起点，对周围构成的三元组与新入的item embedding求Attention求和，表示第一轮扩散之后的用户兴趣，构成第一轮的用户embedding。最后对各轮的用户embedding求和，在于新入的item embedding求相似度
- 优化时用了一个重构loss来限制Item和entity embedding，这也是最大似然估计中的知识库的后验概率
- <img src="https://upload-images.jianshu.io/upload_images/4155986-ec734bf5b72d6bb8.png?imageMogr2/auto-orient/strip|imageView2/2/format/webp" alt="img" style="zoom: 67%;" />



#### MKR（交替优化用户和item的embedding）

- 在推荐系统与知识图谱关联较小时，推荐系统中的物品只有部分和知识图谱中的实体重合，因此用多任务学习的框架
- 两个任务：预测根据用户和item预测CTR，根据head和relation预测tail向量
- 这里交替训练，也即轮流固定一个模型的参数训练另一个。两个模型通过cross feature单元共享同一个对象的信息（item和head）

- <img src="https://upload-images.jianshu.io/upload_images/4155986-1bd15d306b436245.png?imageMogr2/auto-orient/strip|imageView2/2/w/1200/format/webp" alt="img" style="zoom: 50%;" />

### 强化学习用于推荐系统

#### DRN

- 新闻推荐领域中，传统的机器学习模型有几点困难
  - 新闻推荐的时效性和动态变化难以处理，建模时应该也考虑长期回报
  - 没有利用点击之外的用户反馈，文中引入了用户返回APP的时间
  - 现行exploration策略，包括e-greedy和UCB都有一定影响，应该引入更有效的策略
- ![img](https://upload-images.jianshu.io/upload_images/4155986-dea44a2b28da4d40.png?imageMogr2/auto-orient/strip|imageView2/2/format/webp)
- 这里push指的是用户发送请求，agent输入用户和候选集进行推荐，输出是exploitation，exploration结果的结合
  - 用户的点击是反馈，一段时间之后，评估两个策略Q和$\hat{Q}$哪个更好，只优化较差的那个（minor update）
  - 一定时期后，进行经验回放，结合Click和activeness对exploitation模型进行更新（major update）
- state用用户特征和上下文信息表示，action用新闻特征和交互特征表示，Q模型输出当前state采取这个action的预测Q值
- Q的计算包括了当前奖励和未来的折现，奖励包括点击奖励和活跃度奖励
  - 具体一个优化如同一般的Double-DQN，包括两组参数，即交替用长期最大回报的预测短期最大回报计算长期回报
  - 活跃度是本文提出的用作推荐系统反馈的指标，可以理解为使用app的频率，每次点击可以增加活跃度，而活跃度随时间衰减，这是一个0到1之间的数值

#### LIRD（京东的商品推荐）

- 是18年初京东的文章，介绍了强化学习在商品推荐中的应用

- 这里的推荐是List-wise的，即一个action就是推荐一个列表而非一个物品，即计算一整个推荐列表的Q-value，这样既给用户提供更多的多样性，也可以充分考虑物品的相关性

- 利用Actor-Critic架构，即Actor每次根据state输出一个action，Critic评估这个对的Q-value

  <img src="../../../AppData/Roaming/Typora/typora-user-images/image-20210410013119901.png" alt="image-20210410013119901" style="zoom: 67%;" />

- Simulator：利用用户 的历史行为数据训练强化学习系统，数据是非常稀疏的，因此需要一个仿真器来仿真没出现的state-action的reward值。这个仿真器并不是复杂的模型，而是一个对历史数据的加权平均，根据新state和新action与历史数据中各个对余弦距离的计算，以此为权重平均他们的reward（有一些分组近似加速计算的优化）

- Actor：直接用了embedding乘法作为得分，得分最高的K个作为结果，得分排序也作为推荐顺序

#### 京东利用负反馈进行Pair-wise的强化学习

- 《Recommendations with Negative Feedback via Pairwise Deep Reinforcement Learning》是京东在KDD18上的paper
- 对DQN的模型，利用两个GRU处理点击购买序列和忽略序列，即正反馈和负反馈，以此为state，将action定义为只推荐一个物品时推荐的物品（较小的动作空间使其不需要Policy Gradient）

#### 阿里优化搜索广告参数

- 搜索广告需要平衡平台，用户和广告主三方的利益

  <img src="https://i.loli.net/2021/04/10/8y9pbsSLcVqKU2I.png" alt="image-20210410020438807" style="zoom: 67%;" />

  即平台关心广告收入，用户关心点击率和转化率，广告主关心收益。这里加权的参数可以是固定的，但是基于上下文计算最佳的个性化参数，就能更好地导向各方真实的收益。（如商家各个商品利润率不同，各行业价格弹性不同，用户可能给出其他反馈等）

- 来自阿里18年的论文，强化学习的目标是基于s找到最优的a，这里a是排序公式的参数，s是各种上下文。

- Reward设计为校准后的CTR乘上一系列反馈的奖励，即根据展示位置对CTR反馈进行调整

- <img src="https://upload-images.jianshu.io/upload_images/4155986-9e7039237a6bc00e.png?imageMogr2/auto-orient/strip|imageView2/2/format/webp" alt="img" style="zoom: 50%;" />

- 线上探索策略：离线仿真环境和线上环境还是有一定的差距，因此需要在线进行更新，这里用高斯噪声对参数进行扰动，得到n组参数，然后根据扰动方向乘以得到的奖励进行更新，这是一种免梯度的更新方式，更高效。

  <img src="https://i.loli.net/2021/05/01/sXVcANKFWMhIlLE.png" alt="image-20210501052941756" style="zoom: 67%;" />



