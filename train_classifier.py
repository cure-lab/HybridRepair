import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

import random
import numpy as np
import mymodels
from datetime import date

today = date.today()
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print ("Use device: ", device)

data_segmentation = {
    'emnist': { 'TC_end':5e5, 'DC_L_end':5e5+1e4,'DC_U_end':5e5+1e4+2e5}
}

img_channels = {
    'cifar10':3,
    'svhn': 3,
    'emnist': 1,
    'stl10': 3,
    'gtsrb': 3,
}

start_epoch = 0  # start from epoch 0 or last checkpoint epoch
train_acc = 0.0
best_acc = 0.0

# train   
def train(epoch, net, trainloader, criterion, optimizer):
    global train_acc
    net.train()
    total = 0
    correct = 0
    train_loss = 0
    
    for batch_idx, data in enumerate(trainloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs, _ = net(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)

        correct += predicted.eq(labels).sum().item()
        train_loss += loss.item()
        total += labels.size(0)
 
        print ("Epoch [%d] Train Batch [%d/%d]"%(epoch, batch_idx, len(trainloader)), 'Loss: %.3f | Acc: %.3f(%d/%d)'
                                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    train_acc = 100.*correct/total


# test
def test(epoch, net, testloader, criterion, model_save_path):
    global best_acc
    net.eval()
    total = 0
    correct = 0
    test_loss = 0
    for batch_idx, data in enumerate(testloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs, _ = net(inputs)
        loss = criterion(outputs, labels)

        _, pred = outputs.max(1)
        total += labels.size(0)
        test_loss += loss.item()

        correct += pred.eq(labels).sum().item()

        print ("Test Batch [%d/%d]"%(batch_idx, len(testloader)), 'Loss: %.3f | Acc: %.3f(%d/%d)'
                                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    test_acc = 100.*correct/total
    if test_acc > best_acc and (model_save_path is not None):
        print('Saving..')
        if device == 'cuda':
            state = net.module.state_dict()
        else:
            state = net.state_dict()
        state = {
            'net': state,
            'acc': test_acc,
            'epoch': epoch,
        }
        torch.save(state, model_save_path)
        best_acc = test_acc
    return test_acc


def set_weights_for_classes(dataset, weight_per_class):                                                                           
    # weight_per_class = np.random.rand(nclasses)
    print ("weight per class: ", weight_per_class)                                                 
    weight = [0] * len(dataset)     
    for idx, (img, label) in enumerate(dataset):    
        # print ('assign weigh {} / {}'.format(idx, len(dataset)))                                      
        weight[idx] = weight_per_class[label]                                  
    return weight  


def main():
    global data_segmentation

    parser = argparse.ArgumentParser("Train a classifier.")
    parser.add_argument("--dataset", type=str, default=None, 
                        help="supportive dataset: (cifar10, stl10, gtsrb, svhn)")
    parser.add_argument("--model", type=str, default=None, 
                        help="(resnet18, resnet50)")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers")
    # parser.add_argument("--img_size", type=int, default=224, help="batch size")
    parser.add_argument("--n_epochs", type=int, default=200, 
                        help="number of epochs of training")
    parser.add_argument("--class_weight", type=int, default=0, 
                         help='choose from three different settings: all ones, random 1, random 2')
    parser.add_argument("--pretrained", action="store_true", default=False, 
                        help="if we are to use the imagenet pretrained model or not")
    parser.add_argument("--shadow", action='store_true', help='train a shadow classifier with test set')
    parser.add_argument("--data_root", type=str, help="dataset directory")
    parser.add_argument("--save_path", type=str, help="log save directory")
    parser.add_argument('--manualSeed', type=int, default=None, help='manual seed')
    # parser.add_argument("--robust", action='store_true', help='whether apply adversarial training during training')
    args = parser.parse_args()
    print (args)

    # Give a random seed if no manual configuration
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    if device == 'cuda':
        torch.cuda.manual_seed(args.manualSeed)

    print (args.manualSeed)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    log = open(
        os.path.join(args.save_path,
                        'log_train_classifier_seed_{}.txt'.format(args.manualSeed)), 'w')
                        
    data_dir = os.path.join(args.data_root, args.dataset)
    """
    Prepare data and data loader, data will be downloaded automatically to the dataset directory
    """
    # num of classes
    num_class_config = {'stl10':10, 'cifar10':10, 'cifar100':100, 'gtsrb':43, 'tinyimagenet':200, 'svhn':10, 'emnist':47}
    num_classes = num_class_config[args.dataset]

    # weight per class
    if args.dataset == 'cifar10' or args.dataset == 'svhn' or args.dataset == 'stl10':
        weight_per_class_1ist = [np.ones(num_classes), 
                             np.array([0.77132064, 0.02075195, 0.63364823, 0.74880388, 0.49850701,
                                    0.22479665, 0.19806286, 0.76053071, 0.16911084, 0.08833981]), 
                             np.array([0.68535982, 0.95339335, 0.00394827, 0.51219226, 0.81262096,
                                    0.61252607, 0.72175532, 0.29187607, 0.91777412, 0.71457578])]
    elif args.dataset == 'gtsrb':
        weight_per_class_1ist = [np.ones(num_classes), 
                                np.array([0.77216212, 0.48228356, 0.30406492, 0.91053677, 0.26909998,
                                    0.28909198, 0.35131627, 0.97013941, 0.86837562, 0.98054692,
                                    0.45033286, 0.94458961, 0.23112605, 0.55372691, 0.5409784 ,
                                    0.29452659, 0.92728947, 0.52738229, 0.43067916, 0.99193101,
                                    0.28424188, 0.09312185, 0.2133312 , 0.54796805, 0.60172179,
                                    0.7000594 , 0.83156195, 0.86163102, 0.20134217, 0.13067   ,
                                    0.17746337, 0.93035245, 0.0492987 , 0.7938788 , 0.82524376,
                                    0.31146463, 0.99611558, 0.79276013, 0.20267169, 0.61159357,
                                    0.38950843, 0.43659622, 0.53056465]),
                                np.array([0.51164872, 0.04032396, 0.55956398, 0.22991837, 0.14744547,
                                    0.23634622, 0.9156621 , 0.69681508, 0.28457819, 0.69327906,
                                    0.1678602 , 0.80257126, 0.38308791, 0.75513418, 0.12873831,
                                    0.0545106 , 0.43259534, 0.22819396, 0.54183765, 0.16439994,
                                    0.08213382, 0.14668465, 0.3599913 , 0.81741937, 0.3417227 ,
                                    0.51652706, 0.03966325, 0.37644059, 0.87615667, 0.84230859,
                                    0.99723261, 0.14831995, 0.17685309, 0.62291064, 0.46543785,
                                    0.9342646 , 0.28247046, 0.87653412, 0.46566508, 0.16578366,
                                    0.92455307, 0.39519726, 0.32898436])
                                ]   
    else:
        raise ValueError("Dataset not supported by current version")

    weight_per_class = weight_per_class_1ist[args.class_weight]

    # ---- load datasets -----
    if args.dataset == "svhn":
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        val_transform = transforms.Compose(
            [transforms.Resize((32, 32)),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
             ])
        
        # train and val set for classifier are both coming from the offical train set 
        train_set = torchvision.datasets.SVHN(root=data_dir, 
                                                split = 'train',
                                                transform=train_transform, 
                                                download=True)        
        val_set = torchvision.datasets.SVHN(root=data_dir, 
                                                split='train',
                                                transform=val_transform, 
                                                download=True)
        # print (min(train_set.labels), max(train_set.labels))
        num_train = len(train_set)
        indices = list(range(num_train))
        train_idx, valid_idx = indices[:5000], indices[5000:10000]

        train_set = torch.utils.data.Subset(train_set, train_idx)
        val_set = torch.utils.data.Subset(val_set, valid_idx)
        print (len(train_set), len(val_set))

        weights = set_weights_for_classes(train_set, weight_per_class)
        train_index = torch.utils.data.sampler.WeightedRandomSampler(weights=torch.DoubleTensor(weights), 
                                                                    replacement=False,
                                                                    num_samples=int(0.75*len(weights)))
        train_index = list(train_index)

        targets = [i[1] for i in train_set]
        plt.figure()
        plt.hist([targets[i] for i in train_index])
        plt.savefig(os.path.join(args.save_path, 'trainset_distribution.png'))
        
        train_set = torch.utils.data.Subset(train_set, train_index)
        # print ("trainset information: ", len(train_set), train_set[0])
        weights = set_weights_for_classes(val_set, weight_per_class)
        val_index = torch.utils.data.sampler.WeightedRandomSampler(weights=torch.DoubleTensor(weights), 
                                                                    replacement=False,
                                                                    num_samples=int(0.75*len(weights)))
        val_index = list(val_index)

        targets = [i[1] for i in val_set]
        plt.figure()
        plt.hist([targets[i] for i in val_index])
        plt.savefig(os.path.join(args.save_path, 'valset_distribution.png'))

        val_set = torch.utils.data.Subset(val_set, val_index) 
        print_log("dataset segmentation, train/val {}/{}".format(len(train_set), len(val_set)), log)

        train_loader = torch.utils.data.DataLoader(train_set, 
                                    shuffle=True,
                                    batch_size=args.batch_size)
        valid_loader = torch.utils.data.DataLoader(
                                    val_set, 
                                    shuffle=False, batch_size=args.batch_size)

         
    elif args.dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]

        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        val_transform = transforms.Compose(
            [transforms.Resize((32, 32)),
             transforms.ToTensor(),
             transforms.Normalize(mean, std)])
             
        train_set = torchvision.datasets.CIFAR10(root=data_dir, 
                                                train=True, 
                                                transform=train_transform, 
                                                download=True)        
        val_set = torchvision.datasets.CIFAR10(root=data_dir, 
                                                train=True, 
                                                transform=val_transform, 
                                                download=True)
        print (train_set[0])
        num_train = len(train_set)
        indices = list(range(num_train))
        train_idx, valid_idx = indices[:15000], indices[15000:20000]

        train_set = torch.utils.data.Subset(train_set, train_idx)
        val_set = torch.utils.data.Subset(val_set, valid_idx)

        weights = set_weights_for_classes(train_set, weight_per_class)
        train_index = torch.utils.data.sampler.WeightedRandomSampler(weights=torch.DoubleTensor(weights), 
                                                                    replacement=False,
                                                                    num_samples=int(0.75*len(weights)))
        train_index = list(train_index)

        targets = [i[1] for i in train_set]
        plt.hist([targets[i] for i in train_index])
        plt.savefig(os.path.join(args.save_path, 'trainset_distribution.png'))

        train_set = torch.utils.data.Subset(train_set, train_index)
        print ("trainset information: ", len(train_set), train_set[0])
        weights = set_weights_for_classes(val_set, weight_per_class)
        val_index = torch.utils.data.sampler.WeightedRandomSampler(weights=torch.DoubleTensor(weights), 
                                                                    replacement=False,
                                                                    num_samples=int(0.75*len(weights)))
        val_index = list(val_index)

        targets = [i[1] for i in val_set]
        plt.hist([targets[i] for i in val_index])
        plt.savefig(os.path.join(args.save_path, 'valset_distribution.png'))

        val_set = torch.utils.data.Subset(val_set, val_index) 
        print_log("dataset segmentation, train/val {}/{}".format(len(train_set), len(val_set)), log)

        train_loader = torch.utils.data.DataLoader(train_set, 
                                    shuffle=True,
                                    batch_size=args.batch_size)
        valid_loader = torch.utils.data.DataLoader(
                                    val_set, 
                                    shuffle=False, batch_size=args.batch_size)

    elif args.dataset == 'stl10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2471, 0.2435, 0.2616)
        train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        val_transform = transforms.Compose(
            [transforms.Resize((32, 32)),
             transforms.ToTensor(),
             transforms.Normalize(mean, std)])
             
        train_set = torchvision.datasets.STL10(root=data_dir, 
                                                split='train', 
                                                transform=train_transform, 
                                                download=True)        
        val_set = torchvision.datasets.STL10(root=data_dir, 
                                                split='train', 
                                                transform=val_transform, 
                                                download=True)
        print (train_set[0])
        num_train = len(train_set)
        indices = list(range(num_train))
        train_idx, valid_idx = indices[:2000], indices[2000:3000]

        train_set = torch.utils.data.Subset(train_set, train_idx)
        val_set = torch.utils.data.Subset(val_set, valid_idx)

        weights = set_weights_for_classes(train_set, weight_per_class)
        train_index = torch.utils.data.sampler.WeightedRandomSampler(weights=torch.DoubleTensor(weights), 
                                                                    replacement=False,
                                                                    num_samples=int(0.9*len(weights)))
        train_index = list(train_index)

        targets = [i[1] for i in train_set]
        plt.hist([targets[i] for i in train_index])
        plt.savefig(os.path.join(args.save_path, 'trainset_distribution.png'))

        train_set = torch.utils.data.Subset(train_set, train_index)
        print ("trainset information: ", len(train_set), train_set[0])
        weights = set_weights_for_classes(val_set, weight_per_class)
        val_index = torch.utils.data.sampler.WeightedRandomSampler(weights=torch.DoubleTensor(weights), 
                                                                    replacement=False,
                                                                    num_samples=int(0.75*len(weights)))
        val_index = list(val_index)

        targets = [i[1] for i in val_set]
        plt.hist([targets[i] for i in val_index])
        plt.savefig(os.path.join(args.save_path, 'val_distribution.png'))

        val_set = torch.utils.data.Subset(val_set, val_index) 
        print_log("dataset segmentation, train/val {}/{}".format(len(train_set), len(val_set)), log)

        train_loader = torch.utils.data.DataLoader(train_set, 
                                    shuffle=True,
                                    batch_size=args.batch_size)
        valid_loader = torch.utils.data.DataLoader(
                                    val_set, 
                                    shuffle=False, batch_size=args.batch_size)

    elif args.dataset == 'gtsrb':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize([0.3337, 0.3064, 0.3171], [0.2672, 0.2564, 0.2629])
        ])
        test_transform = transforms.Compose(
            [transforms.Resize((32, 32)),
             transforms.ToTensor(),
             transforms.Normalize([0.3337, 0.3064, 0.3171], [0.2672, 0.2564, 0.2629])])

        train_set = torchvision.datasets.ImageFolder(
                                os.path.join(data_dir, 'train'), 
                                transform=train_transform)
        val_set = torchvision.datasets.ImageFolder(
                                os.path.join(data_dir, 'train'), 
                                transform=test_transform)
        num_classes = 43

        shuflle_idxs = np.load('checkpoint/gtsrb/train_idxs.npy')
        train_idx, valid_idx = shuflle_idxs[:2500], shuflle_idxs[2500:5000]

        train_set = torch.utils.data.Subset(train_set, train_idx)
        val_set = torch.utils.data.Subset(val_set, valid_idx)

        weights = set_weights_for_classes(train_set, weight_per_class)
        train_index = torch.utils.data.sampler.WeightedRandomSampler(weights=torch.DoubleTensor(weights), 
                                                                    replacement=False,
                                                                    num_samples=int(0.9*len(weights)))
        train_index = list(train_index)

        targets = [i[1] for i in train_set]
        plt.hist([targets[i] for i in train_index])
        plt.savefig(os.path.join(args.save_path, 'trainset_distribution.png'))

        train_set = torch.utils.data.Subset(train_set, train_index)
        print ("trainset information: ", len(train_set), train_set[0])
        weights = set_weights_for_classes(val_set, weight_per_class)
        val_index = torch.utils.data.sampler.WeightedRandomSampler(weights=torch.DoubleTensor(weights), 
                                                                    replacement=False,
                                                                    num_samples=int(0.75*len(weights)))
        val_index = list(val_index)

        targets = [i[1] for i in val_set]
        plt.hist([targets[i] for i in val_index])
        plt.savefig(os.path.join(args.save_path, 'val_distribution.png'))

        val_set = torch.utils.data.Subset(val_set, val_index) 
        print_log("dataset segmentation, train/val {}/{}".format(len(train_set), len(val_set)), log)

        train_loader = torch.utils.data.DataLoader(train_set, 
                                    shuffle=True,
                                    batch_size=args.batch_size)
        valid_loader = torch.utils.data.DataLoader(
                                    val_set, 
                                    shuffle=False, batch_size=args.batch_size)

    else:
        print_log ("Not valid datasets inputs, the available choice is stl10, imagenet.", log)
    
    
    # save the biased sample index 
    dataset_save_path = os.path.join('./checkpoint', 
                                        args.dataset, 
                                        'ckpt_bias', 
                                        'biased_dataset',
                                        str(args.class_weight))
    if not os.path.exists(dataset_save_path):
        os.makedirs(dataset_save_path)
    np.save(os.path.join(dataset_save_path, 'train.npy'), np.array(train_index))
    np.save(os.path.join(dataset_save_path, 'val.npy'), np.array(val_index))

    # ---- create model -----
    model_names = [name for name in mymodels.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(mymodels.__dict__[name])]
    
    print ('available models: ', model_names)
    print ("current model: ", args.model)
    
    net = mymodels.__dict__[args.model](channels=img_channels[args.dataset], num_classes=num_classes).to(device)
    if device == 'cuda':
        net = nn.DataParallel(net)
    print (net)

    # ----- Train classifer  ------
    # criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    # IP vendor train the classifer
    model_save_path = os.path.join('./checkpoint', args.dataset, 'ckpt_bias')
 
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    learning_rate = 0.1
    # test(0, net, valid_loader, criterion, os.path.join(model_save_path, args.model + '_' + str(args.class_weight) +'_b.t7'))
    # exit()

    for epoch in range(args.n_epochs):
        print(
        '\n==>> [Epoch={:03d}/{:03d}] '.format(epoch, args.n_epochs) \
        + ' [Best : Accuracy={:.2f}]'.format(best_acc ))

        if epoch > 100:
            learning_rate = 0.01
        elif epoch > 150:
            learning_rate = 0.001
        
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
        train(epoch, net, train_loader, criterion, optimizer)
        test(epoch, net, valid_loader, criterion, os.path.join(model_save_path, args.model + '_' + str(args.class_weight) +'_b.t7'))
    
    print_log('save model of ip vendor to' + model_save_path, log)


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


if __name__ == "__main__":
    main()