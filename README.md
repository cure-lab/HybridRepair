# HybridRepair

This is an implementation of our paper "HybridRepair: Towards Annotation-Efficient Repair for Deep Learning Models", which will be presented in ISSTA'22.

If you find this repository useful for your work, please consider citing it as follows:
```
@article{yu2022hybridrepair,
  title={HybridRepair: Towards Annotation-Efficient Repair for Deep Learning Models},
  author={Yu Li, Muxi Chen, and Qiang Xu},
  journal={ISSTA},
  year={2022}
}
```

## Abstract

A well-trained deep learning (DL) model often cannot achieve expected performance after deployment due to the mismatch between the distributions of the training data and the field data in the operational environment. Therefore, repairing DL models is critical, especially when deployed on increasingly larger tasks with shifted distributions. 

Generally speaking, it is easy to obtain a large amount of field data. Existing solutions develop various techniques to select a subset for annotation and then fine-tune the model for repair. While effective, achieving a higher repair rate is inevitably associated with more expensive labeling costs. To mitigate this problem, we propose a novel annotation-efficient repair solution for DL models, namely **HybridRepair**, wherein we take a holistic approach that coordinates the use of a small amount of annotated data and a large amount of unlabeled data for repair. Our key insight is that **accurate yet sufficient** training data is needed to repair the corresponding failure region in the data distribution. Under a given labeling budget, we selectively annotate some data in each failure region and propagate their labels to the neighboring data on the one hand. On the other hand, we take advantage of the semi-supervised learning (SSL) techniques to further boost the training data density. However, different from existing SSL solutions that try to use all the unlabeled data, we only use a selected part of them considering the impact of distribution shift on SSL solutions. 
Experimental results show that HybridRepair outperforms both state-of-the-art DL model repair solutions and semi-supervised techniques for model improvements, especially when there is a distribution shift between the training data and the field data. 

## Getting Started
### Requirements

Install the environment by:
```
conda create -n hybridrepair python=3.6.8
conda activate hybridrepair
pip install -r requirements.txt
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

### Simple Example

To check the general functionality, you can run the following command:
```
sh baseline.sh
```
It will run baseline method 'MCP' for MobileNet on cifar10 (budget=1%, Model A). This commend takes roughly 1 min. The expected output is "T2 Acc before/after repair: 80.38/80.8".

### To validate the paper’s claims and results: 

**Run HybridRepair on cifar10 dataset and MobileNet**
```
sh repair.sh
```
- It will run HybridRepair for MobileNet on cifar10 (budget=1%, Model A). This commend takes roughly 1 hour. The expected output is "T2 Acc before/after repair: 80.38/83.8".
- For other dataset and model, please change the variables 'DATASET' and 'MODEL' correspondingly. 

**Run a baseline method(MCP) on cifar10 dataset and MobileNet**
```
sh baseline.sh
```
- For other baseline methods, please change the variable 'SOLUTION' correspondingly, i.e, 'gini' 'mcp' 'coreset' 'badge' 'SSLConsistency' 'SSLConsistency-Imp' 'SSLRandom'. 
- For other dataset and model, please change the variables 'DATASET' and 'MODEL' correspondingly. 

**Before validation on other dataset and model**, please run the following command to generate models first. The variables 'DATASET' and 'MODEL' in train.sh should be changed correspondingly. 
```
sh train.sh
```

## Contact
If there are any questions, feel free to send a message to yuli@cse.cuhk.edu.hk or mxchen21@cse.cuhk.edu.hk


