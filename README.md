# HybridRepair

This is the original PyTorch implementation of the work: HybridRepair

## Abstract

A well-trained deep learning (DL) model often cannot achieve expected performance after deployment due to the mismatch between the distributions of the training data and the field data in the operational environment. Therefore, repairing DL models is critical, especially when deployed on increasingly larger tasks with shifted distributions. 

Generally speaking, it is easy to obtain a large amount of field data. Existing solutions develop various techniques to select a subset for annotation and then fine-tune the model for repair. While effective, achieving a higher repair rate is inevitably associated with more expensive labeling costs. To mitigate this problem, we propose a novel annotation-efficient repair solution for DL models, namely \emph{HybridRepair}, wherein we take a holistic approach that coordinates the use of a small amount of annotated data and a large amount of unlabeled data for repair. Our key insight is that \emph{accurate yet sufficient} training data is needed to repair the corresponding failure region in the data distribution. Under a given labeling budget, we selectively annotate some data in each failure region and propagate their labels to the neighboring data on the one hand. On the other hand, we take advantage of the semi-supervised learning (SSL) techniques to further boost the training data density. However, different from existing SSL solutions that try to use all the unlabeled data, we only use a selected part of them considering the impact of distribution shift on SSL solutions. 
Experimental results show that \emph{HybridRepair} outperforms both state-of-the-art DL model repair solutions and semi-supervised techniques for model improvements, especially when there is a distribution shift between the training data and the field data. 

## Getting Started
### Requirements

Install the environment by:
```
conda env create -f environment.yml
```
### Simple Example

To check the general functionality, you need to input your DATA_ROOT in baseline.sh first. Then run the following command:
```
sh baseline.sh
```

## Detailed Description
We provide three sh files:
| sh files      |                                                        |
| ------------- | -------------------------------------------------------| 
| train.sh      | Train a test model for model repair                    |
| baseline      | Use baseline model repair techniques to repair models   | 
| repair        | Use Hybrid Repair to repair models                      |
 
You need to input your DATA_ROOT in all the sh files first. (Notice: If your cifar10 dataset is in "./dataset/cifar10", then you only need to input "./dataset") 

We provide three trained **MobileNet** models on **cifar10** in 'check_point\cifar10\ckpt_bias', and a pretrained feature extraction model in 'check_point\cifar10\ckpt_pretrained_mocov3'. 

**If you want to validate on other dataset and model**, please run the following command. The variables 'DATASET' and 'MODEL' in train.sh should be changed correspondingly. 
```
sh train.sh
```
### To validate the paper’s claims and results: 

**Run HybridRepair on cifar10 dataset and MobileNet**
- For other dataset and model, please change the variables 'DATASET' and 'MODEL' correspondingly. 
```
sh repair.sh
```
**Run a baseline method(MCP) on cifar10 dataset and MobileNet**
- For other baseline methods, please change the variable 'SOLUTION' correspondingly, i.e, 'gini' 'coreset' 'badge' 'SSLConsistency' 'SSLConsistency-Imp' 'SSLRandom'. 
- For other dataset and model, please change the variables 'DATASET' and 'MODEL' correspondingly. 
```
sh baseline.sh
```
