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

First, please make sure you have installed Conda. Then, our environment can be installed by:
```
conda create -n hybridrepair python=3.6.9
conda activate hybridrepair
pip install -r requirements.txt
```
To speed up, we think at least one GPU is required. Using purely CPU would be very slow. We use TITAN NVIDIA XP for experiments, and Cuda=11.5, PyTorch 1.10.2, torchvision 0.11.2. Other latest versions can also works.

## Detailed Description
We provide three bash script files:
| sh files      |                                                        |
| ------------- | -------------------------------------------------------| 
| train.sh      | Train a test model for model repair                    |
| baseline.sh      | Use baseline model repair techniques to repair models   | 
| repair.sh        | Use Hybrid Repair to repair models                      |
 
Our code uses the dataset from package ‘torchvision’ and supports automatically downloading. Dataset will be download to './dataset' automatically by default. If you already have the dataset, or you want to try on custom dataset, you can specify the DATA_ROOT variable in all sh files. (Notice: If your cifar10 dataset is stored in "./dataset/cifar10", then you only need to input "./dataset") 

We provide three trained **MobileNet** models (e.g., the model under repair) on **cifar10** in 'check_point\cifar10\ckpt_bias', and a pretrained feature extraction model in 'check_point\cifar10\ckpt_pretrained_mocov3'.

### Simple Example

To check the general functionality, you can run the following command:
```
sh baseline.sh
```
It will run baseline method 'MCP' for MobileNet on cifar10 (budget=1%, Model A). This commend takes roughly 1 min on 2 TITAN NVIDIA XP GPU. The expected output is "T2 Acc before/after repair: 80.38/80.8". You can expect minor rounding errors due to the difference in hardware. 

### To validate the paper’s claims and results

**Run HybridRepair on cifar10 dataset and MobileNet**
```
sh repair.sh
```
- It will run HybridRepair for MobileNet on cifar10 (budget=1%, Model A). This commend takes roughly 1 hour on 2 TITAN NVIDIA XP GPU. The expected output is "T2 Acc before/after repair: 80.38/83.8".
- For other dataset and model, please change the variables 'DATASET' and 'MODEL' correspondingly. 

**Run a baseline method (MCP) on cifar10 dataset and MobileNet**
```
sh baseline.sh
```
- For other baseline methods used in the paper, please change the variable 'SOLUTION' correspondingly, i.e, 'gini' 'mcp' 'coreset' 'badge' 'SSLConsistency' 'SSLConsistency-Imp' 'SSLRandom'. 
- For other dataset and model used in the paper, please change the variables 'DATASET' and 'MODEL' correspondingly, i.e, 'cifar10' 'svhn' 'gtsrb' and 'MobileNet' 'resnet18' 'ShuffleNetG2'. 

**Before validation on other dataset and model used in the paper**, please run the following command to generate models under repair first. The variables 'DATASET' and 'MODEL' in train.sh should be changed correspondingly, i.e, 'cifar10' 'svhn' 'gtsrb' and 'MobileNet' 'resnet18' 'ShuffleNetG2'. 
```
sh train.sh
```

### Extend HybridRepair to other dataset and model
For new dataset:
1. Train models with the train_classifier.py. In this file, you will modify the "num of classes", "weight per class", and "load dataset" part accordingly.
1. After prepare the model, we add the details about the new dataset in selection.py, including transformation, normalization values, number of classes, and how to break the whole dataset into different subsets.
    - mix_test_set: We combine train set and test set of the original division to a large dataset
    - T2_set: The new test set. After retraining, we evalute the retrained model on this set.
    - raw_test_set: The unlabeled dataset. Selection methods selects data points for labeling from this set.
    - selected_set: The set of data selected by selection methods from raw_test_set.
    - model2test_trainset: The set of initial trained data. Model will be retrained on model2test_trainset and selected_set.
4. Please prepare the MoCov3 model beforehand with the opensource code by MoCoV3. After that, supply the MoCov3 model path.

For new model structure:
1. Implemet new model architecure definition in ./mymodels
2. Add a reference in ./mymodels/init.py
3. Train and test the new model.

## Contact
If there are any questions, feel free to send a message to yuli@cse.cuhk.edu.hk or mxchen21@cse.cuhk.edu.hk


