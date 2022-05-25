import argparse
import os,sys
import numpy as np
import random 
import  matplotlib.pyplot as plt
import copy
from tqdm import tqdm
import time
import math
import torch.distributed as dist
import csv
import torch 
import torchvision
import torch.optim as optim
import torch.nn as nn
from torchvision.utils import save_image,make_grid
from torchvision import transforms
from torch.utils.data.sampler import WeightedRandomSampler
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from sklearn.metrics import pairwise_distances
import mymodels 
from sklearn.manifold import TSNE
from scipy.special import softmax
from sklearn.neighbors import KernelDensity
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from sklearn.cluster import OPTICS, KMeans 
from utils import RecorderMeterFlex
from coverage import *
from baseline.coreset import coreset_selection
from baseline.badge import badge_selection
import augmentations
from mymodels import MoCo_ResNet
from hdbscan import HDBSCAN

# device: gpu or cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_names = sorted(name for name in mymodels.__dict__
                     if not name.startswith("__")
                     and callable(mymodels.__dict__[name]))


################# Options ##################################################
############################################################################
parser = argparse.ArgumentParser("Arguments for sample selection.")
parser.add_argument('--data_path',
                    default='../../../datasets/',
                    type=str,
                    help='Path to dataset')
parser.add_argument('--dataset',
                    type=str,
                    choices=['cifar10', 'cifar100', 'imagenet', 'tiny-imagenet-200', 'svhn', 'stl10', 'mnist', 'emnist', 'gtsrb'],
                    help='Choices: cifar10, cifar100, imagenet, svhn, stl10, mnist, gtsrb.')
parser.add_argument('--batch_size',
                    type=int,
                    default=128,
                    help='Batch size')
parser.add_argument('--num_workers',
                    type=int,
                    default=8,
                    help='Batch size')
# test model
parser.add_argument('--model2test_arch',
                    metavar='ARCH',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnext29_8_64)')

parser.add_argument("--model2test_path", 
                        type=str, 
                        default=None, 
                        help="shadow model of test center (resnet18, resnet50)")

parser.add_argument("--model_number", type=int, 
                         help='choose from three different settings: all ones, random 1, random 2')
   
# test budget
parser.add_argument("--p_budget", 
                        type=int, 
                        default=1, 
                        help='number of test cases to be selected in total (%).')


parser.add_argument("--step", 
                        type=int, 
                        default=100, 
                        help='number of test cases to be selected in each iteration.')

# test config
parser.add_argument("--solution",
                    # choices=['random', 'ours', 'mcp', 'gini', 'coreset', 'failure-coverage'],
                    default='random',
                    help='using gini technique to select test cases')


parser.add_argument("--latent_space_plot", 
                        action='store_true',  
                        help='plot all latent space or not.')

parser.add_argument("--sel_u_data", 
                        action='store_true',  
                        help='select unlabeled data for repairing.')

parser.add_argument("--sel_l_data", 
                        action='store_true',  
                        help='select labeled data for repairing.')
parser.add_argument("--u_weight", 
                        action='store_true',  
                        help='weighting the unlabeled data or not.')
parser.add_argument("--disable_dynamic_u_weight", 
                        action='store_true',  
                        help='weighting the unlabeled data or not.')

parser.add_argument("--save_path", 
                        type=str, 
                        default='', help='the file to store the hit ratios')
parser.add_argument("--retrain",
                       action='store_true',
                       help='If retrain after debug')
parser.add_argument("--retrain_lr",
                       type=float,
                       help='the learning rate in the fine-tuning epochs')
parser.add_argument("--retrain_weightdecay",
                       type=float,
                       help='the weight decay in the fine-tuning epochs')
parser.add_argument("--retrain_epoch",
                       type=int,
                       help='the number of epochs to retrain')
parser.add_argument("--exp",
                       type=int,
                       help='control the experiment block to execute')
parser.add_argument("--lam_ul",
                       type=int,
                       default=100, 
                       help='control the weight between u_loss and l_loss')

parser.add_argument("--fe",
                       type=str,
                       default='model2test', 
                       help='The maximum distance between two samples for one to be considere as in the neighborhood of the other. ')

parser.add_argument("--cluster",
                       type=str,
                       default='dbscan', 
                       help='The maximum distance between two samples for one to be considere as in the neighborhood of the other. ')
parser.add_argument("--avg_clustersize",
                       type=int,
                       default=10, 
                       help='Used to decide number of clusters for labeled data selection. ')

parser.add_argument("--auto_tao", 
                        action='store_true',  
                        help='automatically calculate tao instead of using the given tao.')
parser.add_argument("--avg_tao", 
                        action='store_true',  
                        help='automatically calculate tao as the mean tao.')

parser.add_argument("--tao",
                       type=float,
                       default=0.1, 
                       help='The threshold. Decide which part of the unlabeled data is to be selected. ')
parser.add_argument("--eps",
                       type=float,
                       default=0.001, 
                       help='The maximum distance between two samples for one to be considere as in the neighborhood of the other. ')
parser.add_argument("--min_samples",
                       type=int,
                       default=5, 
                       help='The maximum distance between two samples for one to be considere as in the neighborhood of the other. ')
parser.add_argument("--tcp",
                       type=int,
                       default=55, 
                       help='For mocov3 model')

# random seed
parser.add_argument('--manualSeed', type=int, default=None, help='manual seed')


args = parser.parse_args()

# Give a random seed if no manual configuration
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
np.random.seed(args.manualSeed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(args.manualSeed)
    torch.cuda.manual_seed_all(args.manualSeed)
    # True ensures the algorithm selected by CUFA is deterministic
    # Try one of them
    torch.backends.cudnn.deterministic = True # torch.set_deterministic(True)
    
    # False ensures CUDA select the same algorithm each time the application is run
    torch.backends.cudnn.benchmark = False

def _init_fn(worker_id):
    np.random.seed(int(args.manualSeed) + worker_id)

default_hyperparams = {
    'custom':  {'img_size':32,'num_classes':10, 'channel':3,  'feature_dim':512},
    'svhn':    {'img_size':32,'num_classes':10, 'channel':3, 'feature_dim':512},
    'cifar10': {'img_size':32,'num_classes':10, 'channel':3, 'feature_dim':512},
    'gtsrb':   {'img_size':32,'num_classes':43, 'channel':3,  'feature_dim':512},
    'stl10':   {'img_size':32,'num_classes':10, 'channel':3, 'feature_dim':512},
    
}

dataset_split = {
    'svhn':    {'train_end':5000, 'val_end': 10000, 't2_end': 20000},
    'cifar10': {'train_end':15000, 'val_end': 20000, 't2_end': 25000},
    'stl10':   {'train_end':2000,  'val_end': 3000, 't2_end': 7000},
    'gtsrb':   {'train_end':2500,  'val_end': 5000, 't2_end': 10000} 
}


retrain_epoch = args.retrain_epoch 

###############################################################################
def main():
    global retrain_epoch
    debug = False

    # Init logger6
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    
    if args.sel_u_data and not args.sel_l_data:
        tta_config = 'u' 
    elif args.sel_u_data and args.sel_l_data:
        tta_config = 'combined'
    elif not args.sel_u_data and args.sel_l_data:
        tta_config = 'l'
    else:
        tta_config = 'undef'

    log = open(
        os.path.join(args.save_path,
                     'log_seed{}_no{}_method{}_exp{}_eps{}_mpt{}_fe{}_cluster{}_tao{}_{}_bgt{}_avgcluster{}_lam{}.txt'.format(
                        args.manualSeed, args.model_number, args.solution,
                        args.exp, args.eps, args.min_samples, args.fe, args.cluster, args.tao if args.auto_tao is False else 'auto_tao',
                        tta_config, args.p_budget, args.avg_clustersize, args.lam_ul)), 'w')
    args.log = log

    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')),
              log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()),
              log)
    
    print_log(args, log)

    #####################################################
    # ---- Prepare data and data loader, data will be downloaded automatically to the dataset directory------
    if not os.path.isdir(args.data_path):
        os.makedirs(args.data_path)

    if args.dataset == 'cifar10':
        num_classes = 10
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif args.dataset == 'cifar100':
        num_classes = 100
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif args.dataset == 'svhn':
        num_classes = 10
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif args.dataset == 'emnist':
        num_classes = 47
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif args.dataset == 'imagenet':
        num_classes = 1000
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif args.dataset == 'tiny-imagenet-200':
        num_classes = 200
        mean = [0.480,  0.448, 0.397]
        std = [0.230, 0.226, 0.226]
    elif args.dataset == 'gtsrb':
        num_classes = 43
        mean = [0.3337, 0.3064, 0.3171]
        std = [0.2672, 0.2564, 0.2629]
    elif args.dataset == 'stl10':
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2471, 0.2435, 0.2616]
    else:
        assert False, "Unknow dataset : {}".format(args.dataset)
    
    data_dir = os.path.join(args.data_path, args.dataset)
    img_size = default_hyperparams[args.dataset]['img_size']
    data_split = dataset_split[args.dataset]

    if args.dataset == 'svhn':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(img_size, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        test_transform = transforms.Compose(
            [transforms.Resize((img_size, img_size)),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            
        train_set = torchvision.datasets.SVHN(root=data_dir, 
                                                split = 'train',
                                                transform=train_transform, 
                                                download=True)     
        num_train = len(train_set)
        train_idx = list(range(num_train))[:data_split['train_end']]
        val_idx = list(range(num_train))[data_split['train_end']: data_split['val_end']]
        sub_train_set = torch.utils.data.Subset(train_set, train_idx)   

    elif args.dataset == 'cifar10':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(img_size, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose(
            [transforms.Resize((img_size, img_size)),
             transforms.ToTensor(),
             transforms.Normalize(mean, std)])
            
        train_set = torchvision.datasets.CIFAR10(root=data_dir, 
                                                train=True, 
                                                transform=train_transform, 
                                                download=True)
        num_train = len(train_set)
        train_idx = list(range(num_train))[:data_split['train_end']]
        val_idx = list(range(num_train))[data_split['train_end']: data_split['val_end']]
        sub_train_set = torch.utils.data.Subset(train_set, train_idx)   

    elif args.dataset == 'stl10':
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(img_size, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose(
            [transforms.Resize((img_size, img_size)),
             transforms.ToTensor(),
             transforms.Normalize(mean, std)])
            
        train_set = torchvision.datasets.STL10(root=data_dir, 
                                                split='train', 
                                                transform=train_transform, 
                                                download=True)
        num_train = len(train_set)
        train_idx = list(range(num_train))[:data_split['train_end']]
        val_idx = list(range(num_train))[data_split['train_end']: data_split['val_end']]
        sub_train_set = torch.utils.data.Subset(train_set, train_idx)   
    elif args.dataset == 'gtsrb':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(img_size, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose(
            [transforms.Resize((img_size, img_size)),
             transforms.ToTensor(),
             transforms.Normalize(mean, std)])

        train_set = torchvision.datasets.ImageFolder(
                                os.path.join(data_dir, 'train'), 
                                transform=train_transform)
        train_idx = np.load('checkpoint/gtsrb/train_idxs.npy')
        train_set = torch.utils.data.Subset(train_set, train_idx)
        num_train = len(train_set)
        train_idx = list(range(num_train))[:data_split['train_end']]
        val_idx = list(range(num_train))[data_split['train_end']: data_split['val_end']]
        sub_train_set = torch.utils.data.Subset(train_set, train_idx)   

    else:
        raise ValueError('Invalid dataset')

    #####################################################
    # ---- Test Center: Receive models under test------
    print_log("=> creating model 2 test'{}'".format(args.model2test_arch), log)
    # Init model, criterion, and optimizer
    model2test = mymodels.__dict__[args.model2test_arch](num_classes=num_classes, channels=default_hyperparams[args.dataset]['channel']).to(device)
    # print_log("=> network :\n {}".format(model2test), log)
    # load weights
    checkpoint = torch.load(args.model2test_path, map_location=device)
    model2test.load_state_dict(checkpoint['net'])
    
    # restore the trainset training the model
    dataset_save_path = os.path.join('./checkpoint', 
                                        args.dataset, 
                                        'ckpt_bias', 
                                        'biased_dataset',
                                        str(args.model_number))
    sub_train_index = np.load(os.path.join(dataset_save_path, 'train.npy'))
    sub_val_index = np.load(os.path.join(dataset_save_path, 'val.npy'))

    model2test_trainset = torch.utils.data.Subset(sub_train_set, sub_train_index)

    #####################################################
    # ---- Test Center: Prepare Test Dataset------
    #  get testset and testloader 
    """
    mix_test_set:
    T2_set:
    raw_mix_test_set:
    raw_t2_set:
    raw_test_set:
    """
    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    if args.dataset == 'cifar10':
        # test data : total 30k + 10k = 40k
        train_set = torchvision.datasets.CIFAR10(
                                    root=data_dir,
                                    train=True,
                                    transform=test_transform,
                                    download=False )
        test_set = torchvision.datasets.CIFAR10(root=data_dir, 
                                                train=False,
                                                transform=test_transform,
                                                download=False)
        mix_test_set = torch.utils.data.ConcatDataset([train_set, test_set])
        indices = list(range(len(mix_test_set)))
        # train_indices = np.array(indices[:20000])
        T2_indices = np.array(indices[data_split['val_end']: data_split['t2_end']]) # 1k images

        T2_set = torch.utils.data.Subset(mix_test_set, T2_indices)
        # labeled case number
        rest_indices = np.array(indices[data_split['t2_end']:])
        test_set = torch.utils.data.Subset(mix_test_set, rest_indices)
        
        ################ FOR TTA #############################
        raw_orig_train_set = torchvision.datasets.CIFAR10(root=data_dir,
                                                    train=True,
                                                    transform=None,
                                                    download=False )
        raw_orig_test_set = torchvision.datasets.CIFAR10(root=data_dir, 
                                                    train=False,
                                                    transform=None,
                                                    download=False)
        raw_mix_test_set = torch.utils.data.ConcatDataset([raw_orig_train_set, raw_orig_test_set])
        raw_t2_set = torch.utils.data.Subset(raw_mix_test_set, T2_indices)
        raw_test_set = torch.utils.data.Subset(raw_mix_test_set, rest_indices)

    elif args.dataset == 'stl10':
        # test data :  
        train_set = torchvision.datasets.STL10(
                                    root=data_dir,
                                    split='train',
                                    transform=test_transform,
                                    download=False )
        test_set = torchvision.datasets.STL10(root=data_dir, 
                                                split='test',
                                                transform=test_transform,
                                                download=False)
        mix_test_set = torch.utils.data.ConcatDataset([train_set, test_set])
        indices = list(range(len(mix_test_set)))
        T2_indices = np.array(indices[data_split['val_end']:data_split['t2_end']]) # 1k images

        rest_indices = np.array(indices[data_split['t2_end']:])

        T2_set = torch.utils.data.Subset(mix_test_set, T2_indices)
        test_set = torch.utils.data.Subset(mix_test_set, rest_indices)
        ################ FOR TTA #############################
        raw_orig_train_set = torchvision.datasets.STL10(root=data_dir,
                                                    split='train',
                                                    transform=None,
                                                    download=False )
        raw_orig_test_set = torchvision.datasets.STL10(root=data_dir, 
                                                    split='test',
                                                    transform=None,
                                                    download=False)
        raw_mix_test_set = torch.utils.data.ConcatDataset([raw_orig_train_set, raw_orig_test_set])
        raw_t2_set = torch.utils.data.Subset(raw_mix_test_set, T2_indices)
        raw_test_set = torch.utils.data.Subset(raw_mix_test_set, rest_indices)

    elif args.dataset == 'svhn': 
        # test data: unlabeled 3w, labeled 1w, hold-out test data 53w
        train_set = torchvision.datasets.SVHN(root=data_dir,
                                                split='train',
                                                transform=test_transform,
                                                download=True)
        test_set = torchvision.datasets.SVHN(root=data_dir,
                                                split='test',
                                                transform=test_transform,
                                                download=True)
        # T2_set = torchvision.datasets.SVHN(root=data_dir,
        #                                     split='extra',
        #                                     transform=test_transform,
        #                                     download=True)
        mix_test_set = torch.utils.data.ConcatDataset([train_set, test_set])

        num_test = len(mix_test_set)
        indices = list(range(num_test))
        T2_indices = np.array(indices[data_split['val_end']:data_split['t2_end']]) 
        rest_indices = np.array(indices[data_split['t2_end']:])

        # test_set = torch.utils.data.Subset(mix_test_set, rest_indices)
        T2_set = torch.utils.data.Subset(mix_test_set, T2_indices)
        test_set = torch.utils.data.Subset(mix_test_set, rest_indices)
        ################ FOR TTA #############################
        raw_orig_train_set = torchvision.datasets.SVHN(root=data_dir,
                                                    split='train',
                                                    transform=None,
                                                    download=False )
        raw_orig_test_set = torchvision.datasets.SVHN(root=data_dir, 
                                                    split='test',
                                                    transform=None,
                                                    download=False)
        raw_mix_test_set = torch.utils.data.ConcatDataset([raw_orig_train_set, raw_orig_test_set])
        raw_t2_set = torch.utils.data.Subset(raw_mix_test_set, T2_indices)
        raw_test_set = torch.utils.data.Subset(raw_mix_test_set, rest_indices)

    elif args.dataset == 'gtsrb':

        train_set = torchvision.datasets.ImageFolder(
                                os.path.join(data_dir, 'train'), 
                                transform=test_transform)
        test_set = torchvision.datasets.ImageFolder(
                                os.path.join(data_dir, 'test'), 
                                transform=test_transform)
        train_idx = np.load('checkpoint/gtsrb/train_idxs.npy')
        test_idx = np.load('checkpoint/gtsrb/test_idxs.npy')
        train_set = torch.utils.data.Subset(train_set, train_idx)
        test_set = torch.utils.data.Subset(test_set, test_idx)

        mix_test_set = torch.utils.data.ConcatDataset([train_set, test_set])
        indices = list(range(len(mix_test_set)))
        T2_indices = np.array(indices[data_split['val_end']: data_split['t2_end']]) 

        T2_set = torch.utils.data.Subset(mix_test_set, T2_indices)
        # labeled case number
        rest_indices = np.array(indices[data_split['t2_end']:])
        test_set = torch.utils.data.Subset(mix_test_set, rest_indices)
        
        ################ FOR TTA #############################

        raw_orig_train_set = torchvision.datasets.ImageFolder(
                                os.path.join(data_dir, 'train'), 
                                transform=None)
        raw_orig_test_set = torchvision.datasets.ImageFolder(
                                os.path.join(data_dir, 'test'), 
                                transform=None)

        raw_orig_train_set = torch.utils.data.Subset(raw_orig_train_set, train_idx)
        raw_orig_test_set = torch.utils.data.Subset(raw_orig_test_set, test_idx)

        raw_mix_test_set = torch.utils.data.ConcatDataset([raw_orig_train_set, raw_orig_test_set])
        raw_t2_set = torch.utils.data.Subset(raw_mix_test_set, T2_indices)
        raw_test_set = torch.utils.data.Subset(raw_mix_test_set, rest_indices)
    else:
        print_log("not supported dataset", log)
        exit(0)

    
    ####################### A pre-evaluation ############
    # Accuracy before retrain
    # T2: test cases in deployment
    T2_acc = 0
    # if args.retrain:
    print_log ("# of hold-out test inputs: {}".format(len(T2_set)), log)
    T2_acc = test_acc(T2_set, model2test, plot=False)
    print_log("ACC in Hold-Out dataset: {}".format(round(T2_acc, 2)), log)

    print_log ("# of test set inputs: {}".format(len(test_set)), log)
    testset_acc = test_acc(test_set, model2test, plot=True)
    print_log("ACC in test dataset: {}".format(round(testset_acc, 2)), log)

    
    #####################################################
    # Test model2test use all test cases
    print_log('Get the ground truth', log)

    correct_array, logits = test(test_set, model2test, num_classes=num_classes)
    misclass_array = (correct_array==0).astype(int)
    prob = softmax(logits, axis=1)
    confidence = np.sum(np.square(prob), axis=1)

    
    # ---- Get latent vectors of debug dataset------
    if args.fe == 'model2test':
        test_latents = get_features(test_set, model2test).reshape(len(test_set), -1)  
        train_latents = get_features(model2test_trainset, model2test).reshape(len(model2test_trainset), -1)
    elif args.fe == 'scan':
        pretrained_enc_path = 'checkpoint/{}/ckpt_pretrained_scan'.format(args.dataset)
        test_latents, _ = get_pretrained_features(test_set,  os.path.join(pretrained_enc_path, 'simclr_cifar-10.pth.tar')) 
        train_latents, _ = get_pretrained_features(model2test_trainset,  os.path.join(pretrained_enc_path, 'simclr_cifar-10.pth.tar'))
    elif args.fe == 'mocov3':
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:100%d'%args.tcp,
                                    world_size=1, rank=0)
        pretrained_enc_path = 'checkpoint/{}/ckpt_pretrained_mocov3'.format(args.dataset)
        test_latents, _ = get_mocov3_feature(test_set,  os.path.join(pretrained_enc_path, 'checkpoint.pth.tar')) 
        train_latents, _ = get_mocov3_feature(model2test_trainset,  os.path.join(pretrained_enc_path, 'checkpoint.pth.tar'))
    else:
        raise ValueError
    # T2_latents = get_features(T2_set, model2test).reshape(len(T2_set), -1)  

    # # dimension reduction for visualization
    if args.solution is 'cluster' or args.solution is 'failure-coverage':
        # test_codes = PCA(n_components=100).fit_transform(test_latents)

        test_codes = PCA(test_latents , num_components=100)

        test_codes = TSNE(n_components=2).fit_transform(test_codes)
        plot_2d_scatter(test_codes, misclass_array, save_path=args.save_path, 
                        fig_name='misclassification_{}_fe{}'.format(
                            args.model_number, args.fe)) 

        # t2_codes = PCA(n_components=100).fit_transform(T2_latents)
        # # t2_codes = TSNE(n_components=2).fit_transform(t2_codes)

        clusters = cluster_latents(test_latents[np.nonzero(misclass_array)[0]], test_codes[np.nonzero(misclass_array)[0]], args)
        
    # #################### calculate begining TTA on t2 set ##################
    n_iters = 8
    # plot the relation between tta and correctness
    t2_correct_array, test_logit_features = test(T2_set, model2test, num_classes=num_classes)
    t2_misclass_array = (t2_correct_array==0).astype(int)
    t2_failure_idxes = np.nonzero(t2_misclass_array)[0]
    t2_non_failure_idxes = np.nonzero(t2_misclass_array==0)[0]           

    start_tta_values_var, start_tta_single_entropy, start_tta_mean_entropy = get_tta_values(model2test, raw_t2_set, test_transform, n_iters=n_iters, n_class=num_classes)
    
    if debug:
        plt.figure()
        plt.scatter(start_tta_values_var[t2_failure_idxes], start_tta_single_entropy[t2_failure_idxes], c='r',s=0.5, alpha=0.3)
        plt.scatter(start_tta_values_var[t2_non_failure_idxes], start_tta_single_entropy[t2_non_failure_idxes], s=0.5, alpha=0.3)
        plt.savefig(os.path.join(args.save_path, 'scatter_single_entropy_var_{}_t2set.pdf'.format(args.model_number)))
        plt.close()
        print_log("Sum of the tta_values variance: {}".format(start_tta_values_var.sum()), log)
        print_log("Sum of the tta_values single entropy: {}".format(start_tta_single_entropy.sum()), log)

        plt.figure()
        plt.scatter(start_tta_values_var[t2_failure_idxes], start_tta_mean_entropy[t2_failure_idxes], c='r',s=0.5, alpha=0.3)
        plt.scatter(start_tta_values_var[t2_non_failure_idxes], start_tta_mean_entropy[t2_non_failure_idxes], s=0.5, alpha=0.3)
        plt.savefig(os.path.join(args.save_path, 'scatter_mean_entropy_var_{}_t2set.pdf'.format(args.model_number)))
        plt.close()

        plt.figure()
        plt.scatter(start_tta_single_entropy[t2_failure_idxes], start_tta_mean_entropy[t2_failure_idxes], c='r',s=0.5, alpha=0.3)
        plt.scatter(start_tta_single_entropy[t2_non_failure_idxes], start_tta_mean_entropy[t2_non_failure_idxes], s=0.5, alpha=0.3)
        plt.savefig(os.path.join(args.save_path, 'scatter_mean_entropy_single_entropy_{}_t2set.pdf'.format(args.model_number)))
        plt.close()

        plt.figure()
        plt.hist(start_tta_values_var[t2_failure_idxes], bins=100, label='failures', alpha=0.5)
        plt.hist(start_tta_values_var[t2_non_failure_idxes], bins=100, label='correct', alpha=0.5)
        plt.savefig(os.path.join(args.save_path, 'distribution_var_{}_t2set.pdf'.format(args.model_number)))
        plt.close()

        plt.figure()
        plt.hist(start_tta_single_entropy[t2_failure_idxes], bins=100, label='failures', alpha=0.5)
        plt.hist(start_tta_single_entropy[t2_non_failure_idxes], bins=100, label='correct', alpha=0.5)
        plt.savefig(os.path.join(args.save_path, 'distribution_single_entropy_{}_t2set.pdf'.format(args.model_number)))
        plt.close()

        plt.figure()
        plt.hist(start_tta_mean_entropy[t2_failure_idxes], bins=100, label='failures', alpha=0.5)
        plt.hist(start_tta_mean_entropy[t2_non_failure_idxes], bins=100, label='correct', alpha=0.5)
        plt.savefig(os.path.join(args.save_path, 'distribution_mean_entropy_{}_t2set.pdf'.format(args.model_number)))
        plt.close()

        mix = start_tta_single_entropy*start_tta_values_var
        plt.figure()
        plt.hist(mix[t2_failure_idxes], bins=100, label='failures', alpha=0.5)
        plt.hist(mix[t2_non_failure_idxes], bins=100, label='correct', alpha=0.5)
        plt.savefig(os.path.join(args.save_path, 'distribution_single_entropy_and_var_{}_t2set.pdf'.format(args.model_number)))
        plt.close()

        mix = start_tta_mean_entropy*start_tta_values_var
        plt.figure()
        plt.hist(mix[t2_failure_idxes], bins=100, label='failures', alpha=0.5)
        plt.hist(mix[t2_non_failure_idxes], bins=100, label='correct', alpha=0.5)
        plt.savefig(os.path.join(args.save_path, 'distribution_mean_entropy_and_var_{}_t2set.pdf'.format(args.model_number)))
        plt.close()

    # Calculate the density: to see if smooth point have dense training samples
    _, train_logit_features = test(model2test_trainset, model2test, num_classes=num_classes)
    k_neihgbors = 100
    distance_matrix = pairwise_distances(test_logit_features, train_logit_features, metric='cosine')
    sorted_m = np.sort(distance_matrix, axis=1) # ascending
    avg_distance = sorted_m[:, :k_neihgbors].mean(1)

    est_tau = estimate_tau(avg_distance, start_tta_mean_entropy)
    print_log("Estimated tau: {}".format(est_tau), log)
    print_log("average tau: {}".format(start_tta_mean_entropy.mean()), log)

    # relation between variance and density
    if debug:
        plt.figure()
        plt.scatter(start_tta_values_var[t2_failure_idxes], avg_distance[t2_failure_idxes], c='r', s=0.5, alpha=0.3)
        plt.scatter(start_tta_values_var[t2_non_failure_idxes], avg_distance[t2_non_failure_idxes], s=0.5, alpha=0.3)
        plt.savefig(os.path.join(args.save_path, 'density_var_training_{}_t2set.pdf'.format(args.model_number)))
        plt.close()

        # relation between single entropy and density
        plt.figure()
        plt.scatter(start_tta_single_entropy[t2_failure_idxes], avg_distance[t2_failure_idxes], c='r', s=0.5, alpha=0.3)
        plt.scatter(start_tta_single_entropy[t2_non_failure_idxes], avg_distance[t2_non_failure_idxes], s=0.5, alpha=0.3)
        plt.savefig(os.path.join(args.save_path, 'density_single_entropy_training_{}_t2set.pdf'.format(args.model_number)))
        plt.close()

        # relation between mean entropy and density
        plt.figure()
        plt.scatter(start_tta_mean_entropy[t2_failure_idxes], avg_distance[t2_failure_idxes], c='r', s=0.5, alpha=0.3)
        plt.scatter(start_tta_mean_entropy[t2_non_failure_idxes], avg_distance[t2_non_failure_idxes], s=0.5, alpha=0.3)
        plt.savefig(os.path.join(args.save_path, 'density_mean_entropy_training_{}_t2set.pdf'.format(args.model_number)))
        plt.close()

        # from sklearn.neighbors import KernelDensity
        kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(train_latents)
        kde_density = kde.score_samples(test_latents)
        plt.figure()
        plt.scatter(start_tta_mean_entropy[t2_failure_idxes], kde_density[t2_failure_idxes], c='r', s=0.5, alpha=0.3)
        plt.scatter(start_tta_mean_entropy[t2_non_failure_idxes], kde_density[t2_non_failure_idxes], s=0.5, alpha=0.3)
        plt.savefig(os.path.join(args.save_path, 'density_mean_entroy_training_kde_pretrainedfextractor_{}.pdf'.format(args.model_number)))
        plt.close()

    # for p_budget in [ 1, 2, 5, 10, 20]:
    p_budget = args.p_budget
    for p_budget in [p_budget]:
        t_start = time.time()
        budget_total = int(p_budget*len(test_set)/100)
        # step = int(1*len(test_set)/100) # step size = 1%
        step = budget_total # step size = budget
        print_log("Budget: {} (percent) | number {}".format(p_budget, budget_total), log)

        
        print_log("dimension of test latents: {}".format(test_latents.shape), log)
        test_codes=None
        if debug == True:
            test_codes = PCA(test_latents , num_components=100)
            test_codes = TSNE(n_components=2).fit_transform(test_codes)
            plot_2d_scatter(test_codes, misclass_array, save_path=args.save_path, 
                        fig_name='misclassification_{}_fe{}'.format(args.model_number, args.fe)) 
            # # check semantic space
            # testloader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False, num_workers=16)
            # gt_labels = []
            # with torch.no_grad():
            #     for batch_idx, (inputs, labels) in enumerate(testloader):
            #         gt_labels.append(labels)
            # gt_labels = torch.cat(gt_labels)
                    
            # plt.figure()
            # colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
            #             for i in range(10)]

            # for i in range(10):
            #     plt.scatter(test_codes[:, 0][gt_labels==i], test_codes[:, 1][gt_labels==i], c=colors[i], s=0.5) #plot the latents of class 0 to class 19 (20 classes)

            # plt.savefig(os.path.join(args.save_path, 'scan_pretrained_latents.pdf'))
            # plt.close()

        #####################################################
        failure_idxes = np.nonzero(misclass_array)[0]
        non_failure_idxes = np.nonzero(misclass_array==0)[0]

        solution=args.solution
        retrained_model = None
        idx_order = None

        # for label data
        sel_label_data = args.sel_l_data
        if args.auto_tao and args.avg_tao:
            tao = start_tta_mean_entropy.mean()
        elif args.auto_tao:
            tao = est_tau
        else:
            tao = args.tao

        new_test_set = None
        sel_idxes = []
        # for u data
        unlabeled_train_set = None
        u_loss_w = None
        unlabeled_data_selection = args.sel_u_data
        unlabeled_train_set = None

        
        iters = math.ceil(budget_total/step)
        for iter_idx in range(iters):
            if iter_idx != iters - 1:
                budget = step
            else:
                budget = budget_total - step * iter_idx

            print_log('************* Iteration {} Budget in this iter {} ***********'.format(iter_idx, budget), log)

            candidate_sets = []
            ####################### Selection SOLUTION #####################################
            if solution == 'mcp':
                sel_idxes = mcp_selection(logits, num_classes, budget)
            elif solution == 'gini':
                sel_idxes = gini_selection(logits, budget)
            elif solution == 'TTA':
                # calculate the tta value on test set
                """
                
                
                """
                if iter_idx == 0:
                    tta_values, _, tta_mean_entropy = get_tta_values(model2test, raw_test_set, test_transform, n_iters=n_iters)
                else:
                    tta_values, _, tta_mean_entropy = get_tta_values(retrained_model, raw_test_set, test_transform, n_iters=n_iters)

                unsmoothness = tta_mean_entropy

                if iter_idx == 0:
                    plt.figure()
                    failure_hist = np.histogram(unsmoothness[failure_idxes], bins=100, range=(0, 1))
                    non_failure_hist = np.histogram(unsmoothness[non_failure_idxes], bins=100, range=(0, 1))
                    x = failure_hist[1]
                    hist = np.array(non_failure_hist[0]) / (np.array(failure_hist[0] + 1))
                    # print ("x", x[:len(hist)])
                    # print ('hist', hist)
                    plt.bar(x[:len(hist)], hist, width=0.1, label='correct-failure-ratio')
                    plt.savefig(os.path.join(args.save_path, 'correct-failure-ratio-unsmooth-{}-debugset.pdf'.format(args.model_number)))

                ################### labeled data random baseline ###################
                # sel_idxes = np.random.permutation(len(raw_test_set))[:budget]

                ################### labeled data selection: ours ###################
                idx_order = None    
                if sel_label_data == True:
                    # DBSCAN   
                    all_clusters, centers_idx = cluster_latents(test_latents, test_codes, budget, args=args)
 
                    print_log('number of clusters: {}'.format(len(all_clusters)), log)
                    sizes = sum([len(x) for x in all_clusters])
                    if budget > sizes:
                        test_x = []
                        test_y = []
                        text_ygt = []
                        sel_idxes_t = []
                        idx_order = []
                        # label all samples in cluster
                        for clst in all_clusters:
                            c_indexes = np.array(clst)
                            sel_idxes_t.extend(c_indexes) 
                            test_x.extend([test_set[k][0] for k in list(c_indexes)])
                            test_y.extend([test_set[k][1] for k in list(c_indexes)])
                            idx_order.extend(list(c_indexes))
                            text_ygt.extend([test_set[k][1] for k in list(c_indexes)])
                        # plus samples not in cluster
                        rest_budget = budget - sizes
                        idx_not_in_clusters = list(set(np.arange(len(test_set))) - set(idx_order))
                        random.shuffle(idx_not_in_clusters)
                        rest_sel_idx = idx_not_in_clusters[:rest_budget]

                        sel_idxes_t.extend(rest_sel_idx) 
                        test_x.extend([test_set[k][0] for k in list(rest_sel_idx)])
                        test_y.extend([test_set[k][1] for k in list(rest_sel_idx)])
                        idx_order.extend(list(rest_sel_idx))
                        text_ygt.extend([test_set[k][1] for k in list(rest_sel_idx)])
                    else:
                        test_x, test_y, _, sel_idxes_t, idx_order = unsth_region_plabeling(test_set, unsmoothness, 
                                                                all_clusters, centers_idx, budget, log)
                    print_log('smoothed data points {}'.format(len(test_x) - budget), log)


                    # add previsous selected samples
                    if len(sel_idxes) != 0:
                        test_x.extend([test_set[k][0] for k in list(sel_idxes)])
                        test_y.extend([test_set[k][1] for k in list(sel_idxes)])

                    new_test_set = MyCustomDataset(test_x, test_y)
                    sel_idxes.extend(sel_idxes_t)
                

                    # sel_array = np.zeros((len(raw_test_set),))
                    # sel_array[idx_order] = 1
                    # plot_2d_scatter(test_codes, sel_array, save_path=args.save_path, 
                    #                 fig_name='sel_label_{}_{}'.format(args.model_number, args.exp)) 

                ################### unlabeled data selection ###################

                if unlabeled_data_selection == True:
                    if sel_label_data == True: # remove the labeled data
                        # all data in unsmooth clusters has been assigned a label
                        # flattened_unsmooth_cluster = [item for sublist in unsmooth_clusters for item in sublist]
                        # labeled_idx = unsmooth_data_index[np.array(flattened_unsmooth_cluster).astype(int)]
                        # new algo
                        labeled_idx = idx_order

                        # calculate the prediction plabeling accuracy
                        subdataset = torch.utils.data.Subset(raw_test_set, list(labeled_idx))
                        mean_prediction = get_mean_prediction(model2test, subdataset, preprocess=test_transform, 
                                                                n_iters=n_iters)
                        assert (len(mean_prediction) == len(labeled_idx))
                        u_plabel = np.argmax(mean_prediction, axis=1)
                        u_gtlabel = [test_set[k][1] for k in list(labeled_idx)]
                        uplabel_acc = (np.array(u_plabel) == np.array(u_gtlabel)).sum()/len(u_gtlabel)
                        print_log('orig prediction labeling accuracy on sel indexes: {}'.format(uplabel_acc), log)


                    # Fix step 1: turn ineffective learning into effective learning
                    if args.u_weight:
                        u_loss_w = 1 - unsmoothness
                    else:
                        u_loss_w = np.ones(unsmoothness.shape[0])

                    if args.exp == 0: sel_u_location = (unsmoothness <= tao)
                    else: sel_u_location = (unsmoothness > -1)

                    # unsmooth_test_data_idx = np.nonzero(sel_u_location)[0]
                    u_loss_w[~sel_u_location] = 0
                    print_log("selected unlabeled data: {}/{}".format(sel_u_location.sum(), len(sel_u_location)), log)

                    if sel_label_data is True: u_loss_w[labeled_idx] = 1
                    unlabeled_train_data = torch.utils.data.Subset(raw_test_set, list(np.arange(len(u_loss_w))))

                    # # get the uplabel
                    sel_u_idx_t = np.nonzero(sel_u_location)[0]
                    sel_u_data_t = torch.utils.data.Subset(raw_test_set,  list(sel_u_idx_t))
                    mean_p_t = get_mean_prediction(model2test, sel_u_data_t, preprocess=test_transform, 
                                                            n_iters=n_iters)
                    sel_u_l_t = np.argmax(mean_p_t, axis=1)
                    sel_u_gtl_t = [test_set[k][1] for k in list(sel_u_idx_t)]
                    uplabel_acc = (np.array(sel_u_l_t) == np.array(sel_u_gtl_t)).sum()/len(sel_u_gtl_t)
                    print_log('sel u prediction labeling accuracy: {}'.format(uplabel_acc), log)
                    # exit()

                    u_plabel = np.zeros(len(u_loss_w))
                    if sel_label_data is True: u_plabel[idx_order] = test_y
                    unlabeled_train_set = AugMixDataset(unlabeled_train_data, preprocess=test_transform, 
                                                        pesudo_label=u_plabel,
                                                        n_iters=n_iters) 
                    

                    if debug:
                        sel_array = np.zeros((len(raw_test_set),))
                        sel_array[sel_u_location] = 1
                        print_log("ones in u selection array: {} ".format(sel_array.sum()), log)
                        plot_2d_scatter(test_codes, sel_array, save_path=args.save_path, fig_name='sel_ulabel_{}_{}'.format(args.model_number, args.exp)) 


                print_log('selected unlabeled data {}, labeled data {}, current labeled&smoothed data {}'.format(
                    len(unlabeled_train_set) if unlabeled_train_set is not None else 0, len(sel_idxes),
                    len(new_test_set) if new_test_set is not None else 0), log)


            elif solution == 'SSLRandom':
                # labeled data
                indices = np.arange(len(raw_test_set))
                np.random.shuffle(indices)
                sel_idxes = indices[:budget].astype(int)
                rest_indexes = list(set(np.arange(len(raw_test_set))) - set(sel_idxes))
                # unlabeled data
                if len(rest_indexes) > 0:
                    unlabeled_train_data = torch.utils.data.Subset(raw_test_set, np.array(rest_indexes).astype(int))
                    unlabeled_train_set = AugMixDataset(unlabeled_train_data, preprocess=test_transform, n_iters=n_iters) 
                else:
                    unlabeled_train_set = None

                print_log('ssl random: selected unlabeled data {}, labeled data {}'.format(
                    len(unlabeled_train_set) if unlabeled_train_set is not None else 0, 
                    len(sel_idxes)), log)

            elif solution == 'SSLConsistency':
                tta_values, _, _ = get_tta_values(model2test, raw_test_set, test_transform, n_iters=n_iters)
                # labeled data
                ranked_indices = np.argsort(tta_values)[::-1] #descending order
                sel_idxes = ranked_indices[:budget].astype(int)
                rest_indexes = list(set(np.arange(len(raw_test_set))) - set(sel_idxes))
                # unlabeled data
                if len(rest_indexes) > 0:
                    unlabeled_train_data = torch.utils.data.Subset(raw_test_set, np.array(rest_indexes).astype(int))
                    unlabeled_train_set = AugMixDataset(unlabeled_train_data, preprocess=test_transform, n_iters=n_iters) 
                else:
                    unlabeled_train_set = None

                print_log('ssl consistency: selected unlabeled data {}, labeled data {}'.format(
                    len(unlabeled_train_set) if unlabeled_train_set is not None else 0, len(sel_idxes)), log)

            elif solution == 'SSLConsistency-Imp':
                tta_values, _, _ = get_tta_values(model2test, raw_test_set, test_transform, n_iters=n_iters)
                # labeled data
                ranked_indices = np.argsort(tta_values)[::-1] #descending order
                sel_idxes = ranked_indices[:budget].astype(int)
                test_y = [raw_test_set[k][1] for k in sel_idxes]
                idx_order = sel_idxes

                # unlabeled data
                u_plabel = np.zeros(len(raw_test_set))
                u_plabel[sel_idxes] = test_y
                unlabeled_train_set = AugMixDataset(raw_test_set, preprocess=test_transform, 
                                                    pesudo_label=u_plabel,
                                                    n_iters=n_iters) 


                print_log('ssl consistency: selected unlabeled data {}, labeled data {}'.format(
                    len(unlabeled_train_set) if unlabeled_train_set is not None else 0, len(sel_idxes)), log)

            elif solution == 'coreset':
                sel_idxes = coreset_selection(test_latents, budget)
            elif solution == 'badge':
                if args.model2test_arch == 'MobileNet': emb_dim = 1024
                elif args.model2test_arch == 'ShuffleNetG2': emb_dim = 800
                elif args.model2test_arch == 'resnet18': emb_dim = 512
                else:
                    raise ValueError
                sel_idxes = badge_selection(model2test, test_set, budget, emb_dim)
            elif solution == 'failure-coverage':
                # remove some outlier failures which is not helpful in repairing acc
                import itertools
                purged_failure_index = failure_idxes(np.array(list(itertools.chain.from_iterable(clusters))))
                print ('purged failures cnt: ', len(test_latents[purged_failure_index]))
                sel_fail_idxes = coreset_selection(test_latents[purged_failure_index], budget)
                sel_idxes = purged_failure_index[sel_fail_idxes]
            elif solution == 'cluster':
                sel_fail_idxes = round_robin_selection(clusters, budget)
                sel_idxes = failure_idxes[sel_fail_idxes]

            candidate_sets.append(sel_idxes)
            ################# EVALUATE THE REPAIR ACCURACY ########################

            coverage_topmean, coverage_total_topmean = 0, 0
            p_failure = 0
            print_log("\n Evaluating and reparing...", log)
            for _, lb_idxes in enumerate(candidate_sets):
                # The lb_idxes is a binary array
                # print_log("\n ###### Candidate set id {} ######".format(len(candidate_sets)), log)
                # print_log('visulize selected sample via tsne ...', log)
                sel_binary = np.zeros(len(test_set))
                
                if len(lb_idxes) != 0:
                    sel_binary[np.array(lb_idxes)] = 1
                    failure_cnt = misclass_array[lb_idxes].sum()
                    p_failure = 100.0*failure_cnt/budget
                    print_log("Percentage of failure on selected test cases: %s "%(p_failure), log)
                    # plot_2d_scatter(test_codes, sel_binary, save_path=args.save_path, fig_name='selected_{}_{}_{}_{}'.format(solution, budget, args.model_number, i))
                    
                    # # calculate the coverage value
                    # _, _, _, coverage_topmean = coreset_coverage(test_latents, lb_idxes, log)
                    # _, _, _, coverage_total_topmean = coreset_coverage(total_latents, lb_idxes + list(np.arange(len(test_latents), len(total_latents))), log)
                

                # retrain
                new_T2_acc = 0
                new_T2_accs = []
                mean_entropies = []
                if args.retrain:
                    meters = RecorderMeterFlex(retrain_epoch)
                    if args.solution == 'TTA' and new_test_set is not None:
                        selected_set = new_test_set
                    else:
                        selected_set = torch.utils.data.Subset(test_set, list(lb_idxes)) \
                            if len(lb_idxes) != 0 else []
                    
                    if iter_idx == 0:
                        model2test_temp = copy.deepcopy(model2test)
                        optimizer = optim.SGD(model2test_temp.parameters(), lr=args.retrain_lr, 
                                momentum=0.9, weight_decay=args.retrain_weightdecay)

                        finetuned_ckpt_path = args.model2test_path.split('.t7')[0]
                        finetuned_ckpt_path += '_finetuned.t7'
                        if os.path.exists(finetuned_ckpt_path) is False:
                            for epoch in range(retrain_epoch):
                                # fine-tune on training set
                                # if the fine-tuned version is already exist, then use the fine-tuned version instead
                                retrained_model, _, _, _, _ = retrain_model_under_test(model2test_temp, 
                                                                            optimizer=optimizer, 
                                                                            criterion=nn.CrossEntropyLoss(), 
                                                                            sel_set = [],
                                                                            unlabeled_train_set = None,
                                                                            train_set=model2test_trainset, 
                                                                            replay=True,
                                                                            train_transform=train_transform,
                                                                            num_classes=num_classes
                                                                            )
                            
                                # save fine-tuned model
                                torch.save(retrained_model.state_dict(), finetuned_ckpt_path)
                        else:
                            retrained_model_ckpt = torch.load(finetuned_ckpt_path)
                            model2test_temp.load_state_dict(retrained_model_ckpt)
                            retrained_model = model2test_temp

                    # evaluate after retrain
                    finetuned_acc = test_acc(T2_set, retrained_model)
                    uplabel_accs = []
                    best_acc = 0
                    for epoch in range(retrain_epoch):
                        # if args.sel_l_data and args.sel_u_data are only for TTA method
                        # LabelSmoothingLoss(classes=10, smoothing=0.1)
                        if args.sel_l_data is True or args.solution != 'TTA': 
                            retrained_model, total_loss, total_l_loss, total_u_loss, train_acc = retrain_model_under_test(retrained_model, 
                                                                        optimizer=optimizer, 
                                                                        criterion=LabelSmoothingLoss(classes=num_classes, smoothing=0.1), 
                                                                        sel_set = selected_set,
                                                                        unlabeled_train_set = None,
                                                                        u_loss_w = None,
                                                                        train_set=model2test_trainset, 
                                                                        replay=True,
                                                                        train_transform=train_transform,
                                                                        lam=args.lam_ul,
                                                                        num_classes=num_classes
                                                                        )

                        if args.sel_u_data is True or 'SSL' in args.solution:
                            retrained_model, total_loss, total_l_loss, total_u_loss, train_acc = retrain_model_under_test(retrained_model, 
                                                                        optimizer=optimizer, 
                                                                        criterion=nn.CrossEntropyLoss(), 
                                                                        sel_set = [],
                                                                        unlabeled_train_set = unlabeled_train_set,
                                                                        u_loss_w = u_loss_w,
                                                                        train_set=model2test_trainset, 
                                                                        replay=True,
                                                                        train_transform=train_transform,
                                                                        lam=args.lam_ul,
                                                                        idx_order=idx_order,
                                                                        num_classes=num_classes
                                                                        )
                        # evaluate after retrain
                        new_T2_acc = test_acc(T2_set, retrained_model)
                        new_T2_accs.append(new_T2_acc)

                        print_log("T2 Acc before/after fintune: {}/{}".format(round(T2_acc, 4), round(finetuned_acc, 4)), log)
                        print_log("T2 Acc before/after repair: {}/{}".format(round(finetuned_acc, 4), round(new_T2_acc, 4)), log)
                        
                        end_mean_entropy = np.array(0)
                        if args.solution == 'TTA' and args.sel_u_data and args.u_weight and not args.disable_dynamic_u_weight:
                            _, _, end_mean_entropy = get_tta_values(retrained_model, raw_test_set, test_transform, n_iters=n_iters)
                            mean_entropies.append(end_mean_entropy.sum())
                            u_loss_w = 1 - end_mean_entropy
                            u_loss_w[~sel_u_location] = 0
                            if args.sel_l_data is True: u_loss_w[labeled_idx] = 1

                            meters.update(epoch, total_loss, total_l_loss, total_u_loss, end_mean_entropy.sum(), train_acc, new_T2_acc)
                            print_log("Sum of the end_mean_entropy on debug set after retrain: {}".format(end_mean_entropy.sum()), log)


                        # estimate the prediction pseudo-labeling accuracy
                        if args.solution == 'TTA' and args.sel_u_data and args.sel_l_data:
                            mean_prediction = get_mean_prediction(model2test, subdataset, preprocess=test_transform, 
                                                                    n_iters=n_iters)
                            assert (len(mean_prediction) == len(labeled_idx))
                            u_plabel = np.argmax(mean_prediction, axis=1)
                            uplabel_acc = (np.array(u_plabel) == np.array(u_gtlabel)).sum()/len(u_gtlabel)
                            uplabel_accs.append(uplabel_acc)
                            print_log('prediction plabeling accuracy: {} at epoch {}'.format(uplabel_acc, epoch), log)
                            # weighted prediction plabeling accuracy
                            if args.u_weight and not args.disable_dynamic_u_weight:
                                weight = (1 - end_mean_entropy)[labeled_idx]
                                weighted_acc = (((np.array(u_plabel) == np.array(u_gtlabel)).astype(int))*weight).sum()/sum(weight)
                                print_log('weighted prediction plabeling accuracy: {} at epoch {}'.format(weighted_acc, epoch), log)

                            print_log('avg dynamic prediction plabeling accuracy: {}'.format(1.*sum(uplabel_accs)/len(uplabel_accs)), log)

                    if args.solution == 'TTA' and args.sel_u_data is True and args.u_weight:
                        # accuracy with best mean entropy
                        best_idx = np.argmin(mean_entropies)
                        best_acc = new_T2_accs[best_idx]
                        print_log("Mean entropies: {}".format(mean_entropies), log)
                        print_log("Best T2 acc: {}/{}".format(round(finetuned_acc, 4), round(best_acc, 4)), log)
                    else:
                        best_acc = max(new_T2_accs)
                    print_log("T2 accuracies: {}".format(new_T2_accs), log)
                    # meters.plot_curve(save_path=os.path.join(args.save_path, 'repair_training_curve.pdf'))
                    
                    # visualizationg before and after repair on HO dataset
                    # t2_correct_array_after, _ = test(T2_set, model2test, num_classes=num_classes)
                    # intersection = t2_correct_array & t2_correct_array_after 
                    # plt.figure()
                    # plt.scatter(codes_embedded[:, 0][labels==t2_correct_array], codes_embedded[:, 1][labels==t2_correct_array], c='b', s=0.5) #plot the latents of class 0 to class 19 (20 classes)
                    # plt.scatter(codes_embedded[:, 0][labels==t2_correct_array_after], codes_embedded[:, 1][labels==t2_correct_array_after], c='r', s=0.5) #pl
                    # plt.scatter(codes_embedded[:, 0][labels==intersection], codes_embedded[:, 1][labels==intersection], c='grey', s=0.5) #pl
                    # plt.savefig(os.path.join(args.save_path, 'failure_comparison.pdf'))

                # output and logging
                out_file = os.path.join(args.save_path, '{}_result.csv'.format(args.dataset))
                print_log ("writing output to csv file: {}".format(out_file), log)

                with open(out_file, 'a+') as f:
                    writer = csv.writer(f, delimiter=',')
                    writer.writerow([ 
                                    args.model2test_arch, 
                                    args.model2test_path,
                                    args.retrain_lr, 
                                    args.retrain_weightdecay, 
                                    args.retrain_epoch,
                                    solution,
                                    args.manualSeed,
                                    iter_idx,
                                    args.eps,
                                    args.min_samples,
                                    args.avg_clustersize,
                                    args.fe,
                                    args.cluster,
                                    'Budget',
                                    p_budget,
                                    budget,
                                    'FailureRatio',
                                    p_failure,
                                    "Coverage",
                                    coverage_topmean,
                                    coverage_total_topmean,
                                    'AccCompare',
                                    T2_acc,
                                    finetuned_acc,
                                    new_T2_acc,
                                    best_acc,
                                    'tta',
                                    start_tta_values_var.sum(),
                                    end_mean_entropy.sum(),
                                    start_tta_values_var[start_tta_values_var>0.1].mean(),
                                    time.time() - t_start,
                                    args.sel_l_data,
                                    args.sel_u_data,
                                    ])
                


    print_log('success!', log)
    log.close()

    return 


def hier_region_plabeling(test_set, test_latents, me_values, unsmooth_data_index, unsmooth_clusters, budget, log):
    """
    test_set: the whole debug set
    test_latents: the latents of the whole debug set
    me_values: the mean entropy values for data in the whole debug set
    budget: number of labeling budget
    unsmooth_data_index: the index of samples with higher unsmoothness than tau. (the index is w.r.t. the whole debug set)
    unsmooth_clusters: the cluster results of data in unsmooth_data_index (idx is w.r.t the unsmooth_data_index)

    Return: 4 lists
    test_x: the labeled x, inlcuding smoothed ones
    test_y: the labeled y, including smoothed ones
    test_ygt: the groundtruth y of the data x
    sel_idxes: the indexes of samples with true labels (clustering centers)
    """
    test_x = []
    test_y = []
    text_ygt = []
    sel_idxes = []
    idx_order = []

    # evaluate the smoothness of each cluster
    unsth_cluster = []
    for i, idxes in enumerate(unsmooth_clusters):
        sum_unsmooth = me_values[unsmooth_data_index][idxes].sum()
        unsth_cluster.append(sum_unsmooth)
    unsth_cluster = [i/min(unsth_cluster) for i in unsth_cluster]
    print_log('sorted me values of clusters: {}'.format(sorted(unsth_cluster)[::-1]), log)

    # distribute the label quota according to the cluster me_values
    available = budget
    sorted_cluster_idx = list(np.argsort(unsth_cluster)[::-1])
    for i in range(len(unsth_cluster)):
        current_total_unsmooth = sum([unsth_cluster[i] for i in sorted_cluster_idx])
        max_idx = sorted_cluster_idx.pop(0)
        portion = unsth_cluster[max_idx] / current_total_unsmooth
        portion = int(portion*available)
        if portion < 0 or available == 0:
            break
        # ensure two samples are selected for high unsmooth clusters
        portion = 2 if portion <= 1 and available>=2 and \
                    unsth_cluster[max_idx] > (sum(unsth_cluster)/len(unsth_cluster)) else portion
        print_log("# available {}".format(available), log)
        print_log("max unsmooth{}, current_total_unsmooth {}".format(
                unsth_cluster[max_idx], current_total_unsmooth), log)
        print_log('# points to select: {}'.format(portion), log)
        print_log('# points in current cluster: {}'.format(len(unsmooth_clusters[max_idx])), log)
        if portion >= len(unsmooth_clusters[max_idx]):
            print_log('u idx: {}'.format(unsmooth_clusters[max_idx]), log)
            sel_idx_t = unsmooth_clusters[max_idx]
            sel_idx_t = unsmooth_data_index[sel_idx_t]
            costed = len(sel_idx_t)
            sel_idxes.extend(sel_idx_t) 
            test_x.extend([test_set[k][0] for k in list(sel_idx_t)])
            test_y.extend([test_set[k][1] for k in list(sel_idx_t)])
            idx_order.extend(list(sel_idx_t))
            text_ygt.extend([test_set[k][1] for k in list(sel_idx_t)])
        else:
            c_indexes = np.array(unsmooth_clusters[max_idx])
            relative_idx_testset_t = unsmooth_data_index[c_indexes]
            latents_in_cluster = test_latents[relative_idx_testset_t]
            norm = np.sqrt((latents_in_cluster**2).sum(axis=1))[:, None]
            latents_in_cluster = latents_in_cluster / norm

            clustering_t = KMeans(n_clusters=portion, random_state=0).fit(
                                    latents_in_cluster)

            center_idx = np.argmin(cdist(latents_in_cluster, clustering_t.cluster_centers_, 'euclidean'), axis=0)
            assert (len(center_idx) == portion)

            # relative index in clusters --> relative index in test set
            sel_idx_t = relative_idx_testset_t[center_idx]
            sel_idxes.extend(sel_idx_t) 

            # assign all points in the cluster as the label
            print_log('centers: {}'.format(sel_idx_t), log)
            sel_labels = [test_set[k][1] for k in list(sel_idx_t)]
            print_log('sel_labels: {}'.format(sel_labels), log)
            if len(np.unique(sel_labels)) == 1:
                print_log('assign all points in the cluster the same label', log)
                data_idx = unsmooth_data_index[unsmooth_clusters[max_idx]]
                test_x.extend([test_set[k][0] for k in list(data_idx)])
                test_y.extend([sel_labels[0]]*len(data_idx))
                idx_order.extend(list(data_idx))
                text_ygt.extend([test_set[k][1] for k in list(data_idx)])
                
            else:
                print_log("points in one cluster subject to different labels", log)
                # assign labels according to their distance to the centers
                n_clusters = clustering_t.labels_.max() + 1
                for j in range(n_clusters):
                    sample_idx = np.nonzero(clustering_t.labels_ == j)[0]
                    sample_idx = relative_idx_testset_t[sample_idx]   
                    label = test_set[sample_idx[0]][1]
                    test_x.extend([test_set[k][0] for k in list(sample_idx)])
                    test_y.extend([label for _ in list(sample_idx)])
                    idx_order.extend(list(sample_idx))
                    text_ygt.extend([test_set[k][1] for k in list(sample_idx)])

            costed = portion
            
        available -= costed

        # labeling accuracy of the region plabel
        plabel_acc = (np.array(test_y) == np.array(text_ygt)).sum()/len(test_y)
        print_log('region plabeling accuracy: {}'.format(plabel_acc), log)

    return test_x, test_y, text_ygt, sel_idxes, idx_order


def region_plabeling(test_set, unsmooth_data_index, unsmooth_clusters, cluster_centers, log):
    """
    test_set: the whole debug set
    unsmooth_data_index: the index of samples with higher unsmoothness than tau. (the index is w.r.t. the whole debug set)
    unsmooth_clusters: the cluster results of data in unsmooth_data_index (idx is w.r.t the unsmooth_data_index)

    Return: 4 lists
    test_x: the labeled x, inlcuding smoothed ones
    test_y: the labeled y, including smoothed ones
    test_ygt: the groundtruth y of the data x
    sel_idxes: the indexes of samples with true labels (clustering centers)
    """
    test_x = []
    test_y = []
    text_ygt = []
    sel_idxes = []
    idx_order = []

    for idx in range(len(unsmooth_clusters)):
        # relative index in clusters --> relative index in test set
        c_indexes = np.array(unsmooth_clusters[idx])
        relative_idx_testset_t = unsmooth_data_index[c_indexes]
        center_idx = unsmooth_data_index[cluster_centers[idx]]

        sel_idxes.append(center_idx) 
        sel_label = test_set[center_idx][1]
        
        # assign all points in the cluster as the label
        print_log('assign {} points in the cluster the same label {}'.format(len(c_indexes), sel_label), log)
        test_x.extend([test_set[k][0] for k in list(relative_idx_testset_t)])
        test_y.extend([sel_label]*len(relative_idx_testset_t))
        idx_order.extend(list(relative_idx_testset_t))
        text_ygt.extend([test_set[k][1] for k in list(relative_idx_testset_t)])

        # labeling accuracy of the region plabel
        plabel_acc = (np.array(test_y) == np.array(text_ygt)).sum()/len(test_y)
        print_log('region plabeling accuracy: {}'.format(plabel_acc), log)

    return test_x, test_y, text_ygt, sel_idxes, idx_order


def unsth_region_plabeling(test_set, me_values, all_clusters, cluster_centers, budget, log):
    """
    test_set: the whole debug set
    me_values: the unsmooth value of all test data
    all_clusters: cluster in the whole dataset (idx relative to whole debug dataset)
    cluster_centers: the center idx of the clusters (idx relative to whole debug dataset)

    Return: 4 lists
    test_x: the labeled x, inlcuding smoothed ones
    test_y: the labeled y, including smoothed ones
    test_ygt: the groundtruth y of the data x
    sel_idxes: the indexes of samples with true labels (clustering centers)
    """
    test_x = []
    test_y = []
    text_ygt = []
    sel_idxes = []
    idx_order = []

    # evaluate the smoothness of each cluster
    cluster_me_values = []
    for _, idxes in enumerate(all_clusters):
        sum_unsmooth = me_values[idxes].sum()
        cluster_me_values.append(sum_unsmooth)
    cluster_me_values = [i/min(cluster_me_values) for i in cluster_me_values]
    print_log('sorted me values of clusters: {}'.format(sorted(cluster_me_values)[::-1]), log)

    # distribute the label quota according to the cluster me_values
    sorted_cluster_idx = list(np.argsort(cluster_me_values)[::-1])
    for idx in range(int(budget)):
        cluster_idx = sorted_cluster_idx[idx]
        c_indexes = np.array(all_clusters[cluster_idx])
        center_idx = cluster_centers[cluster_idx]

        sel_idxes.append(center_idx) 
        sel_label = test_set[center_idx][1]
        
        # assign all points in the cluster as the label
        print_log('assign {} points in the cluster the same label'.format(len(c_indexes)), log)
        test_x.extend([test_set[k][0] for k in list(c_indexes)])
        test_y.extend([sel_label]*len(c_indexes))
        idx_order.extend(list(c_indexes))
        text_ygt.extend([test_set[k][1] for k in list(c_indexes)])

        # labeling accuracy of the region plabel
        plabel_acc = (np.array(test_y) == np.array(text_ygt)).sum()/len(test_y)
        print_log('region plabeling accuracy: {}'.format(plabel_acc), log)

    return test_x, test_y, text_ygt, sel_idxes, idx_order

    
def estimate_tau(density, mean_entropy):
    """
    density: N*1, the training data density for each sample 
    mean_entropy: N*1, normalized mean entropy for N samples
    """
    average_density = density.mean()
    # the samples have average density
    density -= average_density
    density = np.abs(density)

    indices = np.argsort(density)[:1000]
    tau = np.array(mean_entropy[indices]).mean()

    return tau


def data_cov_selection(test_latents, failure_idxes, budget, log):
    """
    inputs:
        test_latents: The testing pool (latents vectors) where to select samples for repairing
        failure_idxes: The index of failure cases in testset. The failures could be estimated ones
        budget: number of inputs to select
    output:
        the indexes of selected cases
    """

    non_failure_idxes = list(set(np.arange(len(test_latents))) - set(failure_idxes))
    # evaluate the coverage for different failure ratio
    coverages = []
    for failure_ratio in range(0, 100, 5): # control the failure percentage in the selected set
        cov_ts = []
        sels = []
        for i in range(20): # repeat multiple times for each failure percentage: to take the average
            sel_failure_idxes = random.sample(list(failure_idxes), int(failure_ratio*budget/100))
            sel_non_failure_idxes = random.sample(list(non_failure_idxes), budget - int(failure_ratio*budget/100))
            sel_idxes = sel_failure_idxes + sel_non_failure_idxes
            assert (len(sel_idxes) == budget)
            
            _, _, _, cov_topmean = coreset_coverage(test_latents, sel_idxes, log)

            cov_ts.append(cov_topmean)
            sels.append(sel_idxes)

        coverages.append(sum(cov_ts)/len(cov_ts))

        # if the coverage radius is increasing, then stop there
        if (len(coverages) >=2 and coverages[-1] > coverages[-2]) or failure_ratio == 100:
            # find the set that have the average coverage
            index = np.argmin(np.abs(np.array(cov_ts)-sum(cov_ts)/len(cov_ts)))
            lb_indices = sels[index]
            return lb_indices
        if int((failure_ratio+5)*budget/100) > len(failure_idxes):
            index = np.argmin(np.abs(np.array(cov_ts)-sum(cov_ts)/len(cov_ts)))
            lb_indices = sels[index]
            return lb_indices
    return 


def cluster_latents(latents, test_codes, budget, args):
    """Cluster
    Latents: the input is used for the clustering algorithm
    test_codes: 2 dimensional version of the latents, used for tsne visulization
    """

    if args.cluster == 'dbscan':
        clustering = DBSCAN(eps=args.eps, min_samples=args.min_samples, metric='cosine').fit_predict(latents)
    elif args.cluster == 'optics':
        clustering = OPTICS(max_eps=args.eps, min_samples=args.min_samples, metric='cosine').fit_predict(latents)
    elif args.cluster == 'hdbscan':
        clustering = HDBSCAN(min_cluster_size=args.min_samples, metric='cosine').fit_predict(latents)
    elif args.cluster == 'kmeans':
        # to use cosine distance
        length = np.sqrt((latents**2).sum(axis=1))[:, None]
        latents = latents / length
        clustering = KMeans(n_clusters=args.n_clusters, random_state=0).fit_predict(latents)
    elif args.cluster == 'hybrid':
        length = np.sqrt((latents**2).sum(axis=1))[:, None]
        latents = latents / length

        hdbscan_cluster = HDBSCAN(min_samples=args.min_samples).fit(latents)
        clustering = hdbscan_cluster.labels_

        # clustering = DBSCAN(eps=args.eps, min_samples=args.min_samples, metric='cosine').fit_predict(latents)
        non_noise_points = latents[clustering != -1]
        # clustering without noise points
        print_log("non noise point: {}".format(non_noise_points.shape), args.log)

        # cluster based on the number of non-noise points
        b_t = min(budget, len(non_noise_points))
        n_clusters = max(int(len(non_noise_points)//args.avg_clustersize), b_t)
        kmeans_cluster = KMeans(n_clusters=n_clusters, random_state=0).fit(non_noise_points)
        clustering_t  = kmeans_cluster.labels_
        clustering[clustering != -1] = clustering_t

        # cluster centers
        center_latents = kmeans_cluster.cluster_centers_
        center_idx = np.argmin(cdist(latents, center_latents, 'euclidean'), axis=0)

    else:
        raise ValueError

    # sort the cluster by size 
    n_clusters = clustering.max() + 1
    clusters = []
    for i in range(n_clusters):
        samples = np.nonzero(clustering == i)[0]
        clusters.append(list(samples))
    # clusters.sort(reverse=True, key=lambda x: len(x))

    # visulize the clusters
    plt.figure()
    sizes = [len(x) for x in clusters]
    plt.hist(sizes)
    plt.savefig(os.path.join(args.save_path, 'cluster_size_dis_eps{}_minpts{}_fe{}_cluster{}.pdf'.format(args.eps, 
                            args.min_samples, args.fe, args.cluster)))
    
    print_log("# samples been clustered: {}/{}".format(sum(sizes), len(latents)), args.log)
    print_log("# clusters: {}".format(n_clusters), args.log)

    plt.figure()
    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(len(clusters))]

    if test_codes is not None:
        for i, indexes in enumerate(clusters):
            points = test_codes[indexes]
            plt.scatter(points[:, 0], points[:, 1], c=colors[i], s=0.5)
        plt.savefig(os.path.join(args.save_path,'cluster_vis_eps{}_minpts{}_fe{}_cluster{}.pdf'.format(
            args.eps, args.min_samples, args.fe, args.cluster)))

    return clusters, center_idx


def round_robin_selection(clusters, budget):
    # select by roundrobin
    lb_idxes = []
    j = 0
    while len(lb_idxes) < budget:
        # select a random sample in cluster j
        if len(clusters[j]) > 0:
           pop_index = np.random.randint(len(clusters[j]))
           lb_idxes.append(clusters[j].pop(pop_index)) 
        
        # update the cluster to select
        if j < len(clusters) - 1:
            j += 1
        else:
            j = 0        


    return lb_idxes


def mcp_selection(logits, num_classes, budget):
    import baseline.mcp as mcp
    # Multiple-Boundary Clustering and Prioritization to Promote Neural Network Retraining
    prob = softmax(logits, axis=1)
    dicratio=[[] for i in range(num_classes*num_classes)]
    dicindex=[[] for i in range(num_classes*num_classes)]

    for i in range(len(prob)):
        act=prob[i]
        max_index, sec_index, ratio = mcp.get_boundary_priority(act)#max_index 
        dicratio[max_index*num_classes+sec_index].append(ratio)
        dicindex[max_index*num_classes+sec_index].append(i)
    
    lb_idxes = mcp.select_from_firstsec_dic(budget, dicratio, dicindex, num_classes=num_classes)
    return lb_idxes


def gini_selection(logits, budget):
    # get classification result rank
    prob = softmax(logits, axis=1)
    pred = np.sum(np.square(prob), axis=1)
    ranked_indexes = np.argsort(pred)

    return list(ranked_indexes[:budget])


def get_mean_prediction(model2test, dataset, preprocess, n_iters=32):    
    # apply multiple data augmentation and calculate the mean prediction label
    model2test.eval()
    batch_size = 128
    mean_predictions = []
    augset = AugMixDataset(dataset, preprocess, n_iters) 
    augtest_loader = torch.utils.data.DataLoader(
                    augset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=8,
                    pin_memory=True,
                    worker_init_fn=_init_fn)
    
    with torch.no_grad():
        for i, (all_images, label, idxes) in enumerate(augtest_loader):
            all_logits, _ = model2test(torch.cat(all_images, 0).to(device))
            preds = torch.nn.functional.softmax(all_logits.detach().cpu(), dim=1)
            split_preds = torch.split(preds, all_images[0].shape[0])
            # recat_reds: D = [K * N * C], k is the augmentation times, N is batch size, C is number of classes
            recat_preds = torch.cat([e.unsqueeze(0) for e in split_preds], dim=0) 
            
            # mean prediction
            prob = recat_preds.mean(0).numpy()
            mean_predictions.extend(list(prob))

    return np.array(mean_predictions)


def get_tta_values(model2test, raw_data_set, preprocess, n_class=10, n_iters=32):    
    # apply multiple data augmentation and calculate the variance
    # TO BE TESTED
    model2test.eval()
    batch_size = 128
    tta_values_variance = []
    mean_entropies = []
    single_entropies= []
    augset = AugMixDataset(raw_data_set, preprocess, n_iters) 
    augtest_loader = torch.utils.data.DataLoader(
                    augset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=8,
                    pin_memory=True,
                    worker_init_fn=_init_fn)
    
    with torch.no_grad():
        for i, (all_images, label, _) in enumerate(augtest_loader):
            if i % 100 == 0:
                print("calculate tta values progress {}/{}".format(i, int(len(raw_data_set)/batch_size)))
                print ('input size: ', torch.cat(all_images, 0).shape)

            all_logits, _ = model2test(torch.cat(all_images, 0).to(device))
            preds = torch.nn.functional.softmax(all_logits.detach().cpu(), dim=1)
            split_preds = torch.split(preds, all_images[0].shape[0])
            # recat_reds: D = [K * N * C], k is the augmentation times, N is batch size, C is number of classes
            recat_preds = torch.cat([e.unsqueeze(0) for e in split_preds], dim=0) 
            
            # mean entropy
            prob = recat_preds.mean(0)
            mean_entropy = (-prob*np.log2(prob)).sum(1)
            mean_entropies.extend(list(mean_entropy))

            # entropy
            single_entropy = (-recat_preds[0]*np.log2(recat_preds[0])).sum(1)
            single_entropies.extend(list(single_entropy))

            # variance
            variance = recat_preds.var(0).sum(1)
            tta_values_variance.extend(list(variance))

            if i==0:
                grid = make_grid(torch.cat(all_images, 0), nrow=batch_size, padding=2)
                save_image(grid, fp=os.path.join(args.save_path, 'img_augmentation_{}.png'.format(i)))

        tta_values_variance = np.array(tta_values_variance)
        tta_values_variance /= tta_values_variance.max()
        
        single_entropies = np.array(single_entropies)
        single_entropies /= single_entropies.max()

        mean_entropies = np.array(mean_entropies)
        mean_entropies /= mean_entropies.max()

    return tta_values_variance, single_entropies, mean_entropies


class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix augmentation."""
    def __init__(self, dataset, preprocess, n_iters, pesudo_label=None, no_jsd=False):
        self.dataset = dataset
        self.preprocess = preprocess
        self.no_jsd = no_jsd
        self.n_iters = n_iters
        self.pesudo_label = pesudo_label

    def __getitem__(self, i):
        x, y = self.dataset[i]
        if self.no_jsd:
            return aug(x, self.preprocess), y
        else:
            aug_image = [aug(x, self.preprocess) for i in range(self.n_iters)]
            im_tuple = [self.preprocess(x)] + aug_image
            if self.pesudo_label is None:
                return im_tuple, y, i
            else:
                return im_tuple, self.pesudo_label[i], i


    def __len__(self):
        return len(self.dataset)

def plot_2d_scatter(codes_embedded, labels, save_path, fig_name, cmap=plt.get_cmap("seismic")):
    # visulize with tSNE
    # print('tsne plotting...')
    plt.figure()
    if labels is not None:
        # colormap = np.array(['r', 'g', 'k', 'b'])
        # plt.scatter(codes_embedded[:, 0], codes_embedded[:, 1], s=1, c=labels, cmap=cmap, label='latent space visulization') #plot the latents of class 0 to class 19 (20 classes)
        plt.scatter(codes_embedded[:, 0][labels==0], codes_embedded[:, 1][labels==0], c='grey', s=0.5) #plot the latents of class 0 to class 19 (20 classes)
        plt.scatter(codes_embedded[:, 0][labels==1], codes_embedded[:, 1][labels==1], c='r', s=0.5) #plot the latents of class 0 to class 19 (20 classes)
    else:
        plt.scatter(codes_embedded[:, 0], codes_embedded[:, 1], s=0.5) 
 
    # plt.colorbar()
    plt.legend()
    # plt.title('Correct and Incorrect image latents')
    if not os.path.isdir('figs'):
        os.makedirs('figs')
    # plt.xticks(fontsize=14)
    plt.savefig(os.path.join(save_path, fig_name + '.pdf'))
    plt.close()



def test_acc(testset, model, plot=False):
    batch_size = 256 
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8, worker_init_fn=_init_fn)
    correct = 0
    total = 0
    model.eval()
    incorrect_targets=[]

    # test
    with torch.no_grad():
        for (inputs, labels) in tqdm(testloader):
            # print ("Test: {}/{}".format(batch_idx, int(len(testset)/batch_size)+1))
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs)
            _, pred = outputs.max(1)
            correct += pred.eq(labels).sum().item() 
            total += labels.size(0)
            
            # distribution of the incorrect sampels
            incorrect_targets.extend(labels[~(pred.eq(labels))].cpu().numpy())
            # print ("length of incorrect targets", len(incorrect_targets))
    if plot:
        # print ("incorrect targets", incorrect_targets)
        plt.hist(incorrect_targets)
        plt.savefig(os.path.join(args.save_path, 'incorrect_distribution.png'))
    return 100.0*correct/total



def test(testset, model, num_classes=10):
    batch_size = 256
    testsize = len(testset)
    # print_log("test size %s"%testsize)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=16, worker_init_fn=_init_fn)
    correct = 0
    total = 0
    model.eval()

    # test
    correct_array = np.zeros((testsize, ), dtype=int)
    logits = np.zeros((testsize, num_classes), dtype=float)
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(testloader):
            # print ("Extracting features: {}/{}".format(batch_idx, int(len(testset)/batch_size)+1))
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs)
            _, pred = outputs.max(1)
            logits[(batch_idx)*batch_size: (batch_idx+1)*batch_size] = outputs.cpu().numpy()
            correct_array[(batch_idx)*batch_size: (batch_idx+1)*batch_size] = pred.eq(labels).cpu().numpy().astype(int)
            correct += pred.eq(labels).sum().item() 
            total += labels.size(0)

    return correct_array, logits 


def get_features(testset, model):
    batch_size= 256
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=16)
    model.eval()

    # test
    feature_vector = []
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(testloader):
            if batch_idx % 20 == 0:
                print ("Extracting features: {}/{}".format(batch_idx, int(len(testset)/batch_size)+1))
            inputs = inputs.to(device)
            out,h = model(inputs)
            h = h.squeeze()
            h = h.detach()            
            feature_vector.extend(h.cpu().detach().numpy())
    
    feature_vector = np.array(feature_vector)

    return feature_vector


def get_pretrained_features(test_set, pretrain_path):
    backbone = mymodels.__dict__['resnet18_fe']()
    modelt = ContrastiveModel(backbone=backbone)
    state = torch.load(pretrain_path, map_location='cpu')
    modelt.load_state_dict(state)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=256, 
                        shuffle=False, num_workers=16, worker_init_fn=_init_fn)

    modelt = modelt.cuda()
    modelt.eval()
    features = []
    labels = []
    with torch.no_grad():
        for inputs, targets in testloader:
            fe = modelt(inputs.cuda())

            features.extend(fe.cpu().detach().numpy())
            labels.extend(targets.numpy())    

    features = np.array(features)
    labels = np.array(labels)

    return features, labels


# def get_mocov3_feature(dataset, path):
#     dist.init_process_group(backend='nccl', init_method='tcp://localhost:10088',
#                                 world_size=1, rank=0)
#     model = torch.load(path)
#     model = model.cuda()
#     model.eval()
#     testloader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=256, shuffle=False,
#         num_workers=16, pin_memory=True)

#     features = []
#     labels = []
#     with torch.no_grad():
#         for inputs, targets in testloader:
#             fe = model.module.predictor(model.module.base_encoder(inputs))
#             fe = nn.functional.normalize(fe, dim=1)
#             features.extend(fe.cpu().detach().numpy())
#             labels.extend(targets.numpy())    

#     features = np.array(features)
#     labels = np.array(labels)

#     return features, labels


def get_mocov3_feature(dataset, path):
    import torchvision.models as torchvision_models
    from functools import partial

    
    model = MoCo_ResNet(
            partial(torchvision_models.__dict__['resnet50'], zero_init_residual=True), 
            64, 4096, 1.0)
    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    loader_te = torch.utils.data.DataLoader(
        dataset,
        batch_size=256, shuffle=False,
        num_workers=20, pin_memory=True)
    feature_list = []
    Y_list = []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader_te):
            x, y = x.cuda(), y.cuda()
            feature = model.module.predictor(model.module.base_encoder(x))
            feature = nn.functional.normalize(feature, dim=1)
            feature_list.extend(feature.cpu().detach().numpy())
            Y_list.extend(y.cpu().numpy())

    return np.array(feature_list),np.array(Y_list)

# add mixup augmentation
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def aug(image, preprocess):
    """Perform AugMix augmentations and compute mixture.
    Args:
        image: PIL.Image input image
        preprocess: Preprocessing function which should return a torch tensor.
    Returns:
        mixed: Augmented and mixed image.
    """
    aug_list = augmentations.augmentations
    # if args.all_ops:
    #     aug_list = augmentations.augmentations_all
    mixture_width, mixture_depth = 1, -1
    aug_severity = 3
    ws = np.float32(np.random.dirichlet([1] * mixture_width))
    m = np.float32(np.random.beta(1, 1))

    mix = torch.zeros_like(preprocess(image))
    for i in range(mixture_width):
        image_aug = image.copy()
        depth = mixture_depth if mixture_depth > 0 else np.random.randint(
                1, 4)
        for _ in range(depth):
            op = np.random.choice(aug_list)
            image_aug = op(image_aug, aug_severity)
        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * preprocess(image_aug)

    mixed = (1 - m) * preprocess(image) + m * mix

    return mixed


def retrain_model_under_test(net, sel_set, optimizer, criterion, train_set, 
                                train_transform, unlabeled_train_set, u_loss_w=None, replay=False, lam=100, 
                                idx_order=None, num_classes=10):
    l_batch_size, u_batch_size = 64, 64
    if replay:
        # replay_train_set = torch.utils.data.Subset(train_set, np.arange(len(sel_set)))   
        # obtain the inference result on the training dataset
        correct_array, _ =  test(train_set, net, num_classes=num_classes)
        replay_train_set = torch.utils.data.Subset(train_set, np.nonzero(correct_array)[0])   
        print ('replay {} training sample from {} samples'.format(len(replay_train_set), len(train_set)))
        train_set_temp = torch.utils.data.ConcatDataset([replay_train_set, sel_set])
        recover_loader = torch.utils.data.DataLoader(train_set_temp, batch_size=l_batch_size, 
                            shuffle=True, drop_last=True, num_workers=args.num_workers, worker_init_fn=_init_fn)
        train_len_l = len(train_set_temp)
    else:
        train_len_l = len(sel_set)
        recover_loader = torch.utils.data.DataLoader(sel_set, batch_size=l_batch_size, num_workers=args.num_workers,
                                            shuffle=True, worker_init_fn=_init_fn)
        
    print_log('recover loader length: {}'.format(len(recover_loader)), args.log)

    # unlabeled 
    unlabeled_train_loader = None
    if unlabeled_train_set is not None:
        train_len_u = len(unlabeled_train_set)
        print ('length of unlabeled train data: ', train_len_u)
        w_sampler = WeightedRandomSampler(u_loss_w, len(u_loss_w)) if u_loss_w is not None else None
        unlabeled_train_loader = torch.utils.data.DataLoader(unlabeled_train_set, 
                                    sampler=w_sampler,
                                    num_workers=args.num_workers,
                                    batch_size=u_batch_size, shuffle=False, drop_last=True, worker_init_fn=_init_fn)
        unlabeled_train_iter = iter(unlabeled_train_loader)
        
    net.train()
    total = 0
    correct = 0
    total_loss = 0
    total_l_loss = 0
    total_u_loss = 0

    recover_loader_iter = iter(recover_loader)
    n_batches =  int(train_len_l/l_batch_size) if unlabeled_train_loader is None \
                    else max(int(train_len_l/l_batch_size), int(train_len_u/u_batch_size))
    for batch_idx in range(n_batches):
        # labeled data
        try:
            inputs, labels = recover_loader_iter.next()
        except:
            recover_loader_iter = iter(recover_loader)
            inputs, labels = recover_loader_iter.next()


        targets_x = torch.zeros(l_batch_size, num_classes).scatter_(1, labels.view(-1,1).long(), 1)
        
        if unlabeled_train_set is None:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, _ = net(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            _, predicted = outputs.cpu().max(1)
            correct += predicted.eq(labels.cpu()).sum().item()
            total_l_loss += loss.item()
            total += labels.size(0)


            # print ("Train Batch [%d/%d]"%(batch_idx, len(recover_loader)), 'Loss: %.3f | Acc: %.3f(%d/%d)'
            #                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        else:
            # unlabeled data  and labeled data co-training 
            T = 0.5 # temperature
            try:
                all_images, plabel, idxes = unlabeled_train_iter.next()
            except:
                unlabeled_train_iter = iter(unlabeled_train_loader)
                all_images, plabel, idxes = unlabeled_train_iter.next()
            
            debug = False
            if debug == True:
                for img in all_images:
                    print ('shape of all images:', img.shape)
                # save images for debug 

            if batch_idx == 0 and debug == True:
                print ('save images')
                grid = make_grid(torch.cat([img[0].unsqueeze(0) for img in all_images], 0), nrow=len(all_images), padding=2)
                save_image(grid, fp=os.path.join(args.save_path, 'checkpoint_img.png'))

            
            with torch.no_grad():
                aug_logits, _ = net(torch.cat(all_images, 0).to(device))
                if debug: print ('Logits shape:', aug_logits.shape)
                all_preds = torch.nn.functional.softmax(aug_logits, dim=-1)
                if debug: print ('All preds:', all_preds.shape, all_preds.requires_grad)
                all_preds = torch.split(all_preds, all_images[0].shape[0])
                if debug: print ('All preds:', all_preds[0].shape, all_preds[0].unsqueeze(0).shape)
                all_preds = torch.cat([e.unsqueeze(0) for e in all_preds], dim=0)
                if debug: print ('New all preds:', all_preds.shape, all_preds.requires_grad)

                avg_p = all_preds.mean(0)
                if debug: print ('Avg_p:', avg_p.shape)
                shapened_p = avg_p**(1/T)
                shapened_p = shapened_p / shapened_p.sum(dim=1, keepdim=True)
                shapened_p = shapened_p.detach()
                if debug: print ('Shapened_p:', shapened_p.shape)

            if idx_order is not None:
                intersect_idx = np.in1d(idxes, idx_order).nonzero()[0]
                # shapened_p[intersect_idx] = one_hot(plabel[intersect_idx], n_classes=10).to(device)
                shapened_p[intersect_idx] = smooth_one_hot(true_labels=plabel[intersect_idx], classes=num_classes, smoothing=0.1).to(device)

            # mixup unlabeled data
            n_iters = 8
            all_lu_inputs = torch.cat([inputs] + all_images, 0).to(device)
            
            all_lu_targets = torch.cat([targets_x.to(device)] + [shapened_p.to(device)]*(n_iters+1), dim=0)
            if debug: print ("u shape", all_lu_inputs.shape, all_lu_targets.shape)
            l = np.random.beta(0.75, 0.75)
            l = max(l, 1-l)
            idx = torch.randperm(all_lu_inputs.size(0))
            input_a, input_b = all_lu_inputs, all_lu_inputs[idx]
            # print (idx.shape, all_lu_targets.shape, all_lu_targets[idx])
            target_a, target_b = all_lu_targets, all_lu_targets[idx]

            mixed_input = l * input_a + (1 - l) * input_b
            mixed_target = l * target_a + (1 - l) * target_b


            optimizer.zero_grad()

            logits_lu, _ = net(mixed_input)
            probs_lu = torch.softmax(logits_lu, dim=1)

            # cross entropy loss for labeled data
            l_loss = criterion(logits_lu[:len(labels)], labels.to(device))

            # consistency loss: we use l2

            u_loss = torch.mean((probs_lu[len(labels):] - mixed_target[len(labels):])**2)
            if debug: print ("U loss: ", u_loss.requires_grad)

            
            loss = l_loss + lam*u_loss
            loss.backward()
            optimizer.step()
            total_u_loss += u_loss.item()
            total_l_loss += l_loss.item()

            _, predicted = logits_lu[:len(labels)].cpu().max(1)
            correct += predicted.eq(labels.cpu()).sum().item()
            total += labels.size(0)
            print ("Train Unlabeled Batch [%d/%d]"%(batch_idx, len(unlabeled_train_loader)), 'Loss: %.3f'
                                    % (total_u_loss/(batch_idx+1)))


        total_loss = total_u_loss + total_l_loss
        train_acc = 100.*correct/total

        print ("Train Batch [%d/%d]"%(batch_idx, len(recover_loader)), 'Loss: %.3f | Acc: %.3f(%d/%d)'
                                % (total_l_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return net, total_loss, total_l_loss, total_u_loss, train_acc


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def one_hot(label, n_classes):
    label = torch.tensor(label)
    return torch.zeros(len(label), n_classes).scatter_(1, label.view(-1,1).long(), 1)


def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method

    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=true_labels.device)
        true_dist.fill_(smoothing / (classes - 1))
        true_dist.scatter_(1, true_labels.data.unsqueeze(1).long(), confidence)
    return true_dist


class ContrastiveModel(nn.Module):
    def __init__(self, backbone, features_dim=128):
        super(ContrastiveModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim']
 
        self.contrastive_head = nn.Sequential(
                nn.Linear(self.backbone_dim, self.backbone_dim),
                nn.ReLU(), nn.Linear(self.backbone_dim, features_dim))
    

    def forward(self, x):
        features = self.backbone(x)
        features = self.contrastive_head(features)
        features = F.normalize(features, dim = 1)
        return features


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()



def PCA(X , num_components):
     
    #Step-1
    X_meaned = X - np.mean(X , axis = 0)
     
    #Step-2
    cov_mat = np.cov(X_meaned , rowvar = False)
     
    #Step-3
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
     
    #Step-4
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
     
    #Step-5
    eigenvector_subset = sorted_eigenvectors[:,0:num_components]
     
    #Step-6
    X_reduced = np.dot(eigenvector_subset.transpose() , X_meaned.transpose() ).transpose()
     
    return X_reduced
    
    
class MyCustomDataset(Dataset):
    def __init__(self, X, Y):
        # stuff
        self.X = X
        self.Y = Y
        
    def __getitem__(self, index):
        # stuff
        x, y = self.X[index], self.Y[index]
        return x, y

    def __len__(self):
        return len(self.X) # of how many examples(images?) you have



if __name__ == '__main__':
    main()
