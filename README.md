# HybridRepair

This is the original PyTorch implementation of the work: HybridRepair

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
