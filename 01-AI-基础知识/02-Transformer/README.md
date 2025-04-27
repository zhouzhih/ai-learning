# Transformer架构

Transformer是一种基于自注意力机制的神经网络架构，由Google在2017年的论文《Attention Is All You Need》中提出。它彻底改变了自然语言处理(NLP)领域，并逐渐扩展到计算机视觉等其他领域。

## 目录

- [基本概念](#基本概念)
- [架构组件](#架构组件)
- [工作原理](#工作原理)
- [变体与发展](#变体与发展)
- [应用场景](#应用场景)
- [学习资源](#学习资源)

## 基本概念

### 什么是Transformer？

Transformer是一种完全基于注意力机制的序列转换模型，不依赖循环或卷积结构。它通过自注意力机制直接建模序列中任意位置之间的依赖关系，实现了高效的并行计算和长距离依赖的捕捉。

### 核心优势

- **并行计算**：不同于RNN的顺序处理，Transformer可以并行处理整个序列
- **长距离依赖**：自注意力机制可以直接建模序列中任意位置之间的关系
- **可解释性**：注意力权重提供了模型决策的可视化解释
- **可扩展性**：架构易于扩展到更大规模

## 架构组件

Transformer由编码器(Encoder)和解码器(Decoder)两部分组成，每部分包含多个相同的层。

### 编码器(Encoder)

编码器由N个相同的层堆叠而成，每层包含两个子层：

1. **多头自注意力机制(Multi-Head Self-Attention)**
2. **前馈神经网络(Feed-Forward Neural Network)**

每个子层都使用残差连接(Residual Connection)和层归一化(Layer Normalization)。

### 解码器(Decoder)

解码器也由N个相同的层堆叠而成，每层包含三个子层：

1. **掩码多头自注意力机制(Masked Multi-Head Self-Attention)**
2. **编码器-解码器注意力机制(Encoder-Decoder Attention)**
3. **前馈神经网络(Feed-Forward Neural Network)**

同样，每个子层都使用残差连接和层归一化。

### 其他关键组件

- **位置编码(Positional Encoding)**：为模型提供序列中位置信息
- **嵌入层(Embedding Layer)**：将输入标记转换为向量表示
- **线性层和Softmax**：将解码器输出转换为概率分布

## 工作原理

### 自注意力机制(Self-Attention)

自注意力机制是Transformer的核心，它通过计算序列中每个位置与所有位置的关系来生成表示。

计算步骤：

1. 将输入向量转换为查询(Query)、键(Key)和值(Value)向量
2. 计算查询和键之间的点积，得到注意力分数
3. 对注意力分数进行缩放和Softmax归一化
4. 将归一化后的分数与值向量加权求和

数学表达式：
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

### 多头注意力(Multi-Head Attention)

多头注意力通过并行运行多个自注意力"头"，允许模型关注不同位置的不同表示子空间。

### 前馈网络(Feed-Forward Network)

每个位置的表示独立通过相同的前馈网络，包含两个线性变换和一个ReLU激活函数：
```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```

### 训练过程

Transformer使用教师强制(Teacher Forcing)训练，即在训练过程中使用真实的目标序列作为解码器的输入。

## 变体与发展

### BERT(Bidirectional Encoder Representations from Transformers)

BERT仅使用Transformer的编码器部分，通过掩码语言模型(MLM)和下一句预测(NSP)任务进行预训练，用于各种NLP任务。

### GPT(Generative Pre-trained Transformer)

GPT系列仅使用Transformer的解码器部分，通过自回归语言建模进行预训练，专注于生成任务。

### T5(Text-to-Text Transfer Transformer)

将所有NLP任务统一为文本到文本的转换问题，使用完整的编码器-解码器架构。

### Vision Transformer(ViT)

将Transformer应用于计算机视觉任务，将图像分割为补丁并作为序列处理。

## 应用场景

- **机器翻译**：源语言到目标语言的翻译
- **文本摘要**：生成文档的简洁摘要
- **问答系统**：根据上下文回答问题
- **语言建模**：预测序列中的下一个标记
- **图像生成**：基于文本描述生成图像(如DALL-E)
- **语音识别**：将语音转换为文本

## 学习资源

### 原始论文

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani等人，2017

### 教程和课程

- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Jay Alammar
- [Stanford CS224n: Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/)
- [Hugging Face Transformers课程](https://huggingface.co/course/chapter1/1)

### 开源实现

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [TensorFlow官方实现](https://www.tensorflow.org/text/tutorials/transformer)
- [PyTorch官方实现](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)

### 可视化资源

- [Transformer模型可视化](https://transformer.huggingface.co/)
- [BertViz: BERT注意力可视化](https://github.com/jessevig/bertviz)

### 进阶阅读

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (GPT-3论文)
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) (ViT论文)
