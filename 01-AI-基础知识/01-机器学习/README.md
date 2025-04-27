# 机器学习基础

机器学习是人工智能的一个子领域，专注于开发能够从数据中学习并做出预测或决策的算法和模型，而无需显式编程。

## 目录

- [基本概念](#基本概念)
- [机器学习类型](#机器学习类型)
- [常见算法](#常见算法)
- [评估指标](#评估指标)
- [学习资源](#学习资源)

## 基本概念

### 什么是机器学习？

机器学习是一种使计算机能够在没有明确编程的情况下学习的方法。它专注于开发能够访问数据并使用数据自行学习的算法。

### 核心术语

- **特征(Features)**: 用于训练模型的输入变量
- **标签(Labels)**: 预测目标或输出变量
- **样本(Samples)**: 单个数据点
- **模型(Model)**: 从数据中学习的算法或数学表示
- **训练(Training)**: 模型从数据中学习的过程
- **推理(Inference)**: 使用训练好的模型进行预测

## 机器学习类型

### 监督学习(Supervised Learning)

算法从带标签的训练数据中学习，以预测新数据的标签。

**常见应用**:
- 分类问题(Classification)
- 回归问题(Regression)

### 无监督学习(Unsupervised Learning)

算法从无标签的数据中发现隐藏的模式或内在结构。

**常见应用**:
- 聚类(Clustering)
- 降维(Dimensionality Reduction)
- 异常检测(Anomaly Detection)

### 强化学习(Reinforcement Learning)

算法通过与环境交互并接收反馈(奖励或惩罚)来学习最佳行动策略。

**常见应用**:
- 游戏AI
- 机器人控制
- 自动驾驶

## 常见算法

### 监督学习算法

- **线性回归(Linear Regression)**
- **逻辑回归(Logistic Regression)**
- **决策树(Decision Trees)**
- **随机森林(Random Forests)**
- **支持向量机(Support Vector Machines, SVM)**
- **K-最近邻(K-Nearest Neighbors, KNN)**
- **朴素贝叶斯(Naive Bayes)**
- **神经网络(Neural Networks)**

### 无监督学习算法

- **K-均值聚类(K-Means Clustering)**
- **层次聚类(Hierarchical Clustering)**
- **主成分分析(Principal Component Analysis, PCA)**
- **t-SNE(t-Distributed Stochastic Neighbor Embedding)**
- **DBSCAN(Density-Based Spatial Clustering of Applications with Noise)**

### 强化学习算法

- **Q-Learning**
- **深度Q网络(Deep Q-Network, DQN)**
- **策略梯度(Policy Gradients)**
- **Actor-Critic方法**

## 评估指标

### 分类问题

- **准确率(Accuracy)**
- **精确率(Precision)**
- **召回率(Recall)**
- **F1分数(F1 Score)**
- **ROC曲线和AUC**
- **混淆矩阵(Confusion Matrix)**

### 回归问题

- **均方误差(Mean Squared Error, MSE)**
- **均方根误差(Root Mean Squared Error, RMSE)**
- **平均绝对误差(Mean Absolute Error, MAE)**
- **R²(决定系数)**

## 学习资源

### 在线课程

- [吴恩达机器学习课程](https://www.coursera.org/learn/machine-learning)
- [Stanford CS229: 机器学习](https://cs229.stanford.edu/)
- [Fast.ai 实用机器学习](https://course.fast.ai/)

### 书籍

- 《机器学习实战》- Peter Harrington
- 《机器学习》- 周志华(西瓜书)
- 《统计学习方法》- 李航
- 《Pattern Recognition and Machine Learning》- Christopher Bishop

### 实践平台

- [Kaggle](https://www.kaggle.com/)
- [Google Colab](https://colab.research.google.com/)
- [UCI机器学习资源库](https://archive.ics.uci.edu/ml/index.php)

### 开源库

- [Scikit-learn](https://scikit-learn.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [PyTorch](https://pytorch.org/)
- [XGBoost](https://xgboost.readthedocs.io/)
