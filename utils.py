import os, sys, time, random
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from torch import nn

def piecewise_clustering(var, gamma, beta):
    var1=(var[var.ge(0)]-var[var.ge(0)].mean()).pow(2).sum()
    var2=(var[var.le(0)]-var[var.le(0)].mean()).pow(2).sum()
    val=gamma*var1 + beta*var2
    return val

def clustering_loss(model, lambda_coeff):
    
    pc_loss = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            pc_loss += piecewise_clustering(m.weight, lambda_coeff, lambda_coeff)
    
    return pc_loss 

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class RecorderMeterFlex(object):
    """Computes and stores the minimum loss value and its epoch index"""

    def __init__(self, total_epoch):
        self.reset(total_epoch)
        
    def reset(self, total_epoch):
        assert total_epoch > 0
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_value = np.zeros((self.total_epoch, 4),
                                     dtype=np.float32)  # [epoch, train/val]
        self.epoch_acc = np.zeros((self.total_epoch, 2),
                                     dtype=np.float32)  # [epoch, train/val]

    def update(self, idx, train_loss, l_loss, u_loss, smoothness, train_acc, test_acc):
        assert idx >= 0 and idx < self.total_epoch, 'total_epoch : {} , but update with the {} index'.format(
            self.total_epoch, idx)
        
        self.epoch_value[idx, 0] = train_loss
        self.epoch_value[idx, 1] = l_loss
        self.epoch_value[idx, 2] = u_loss
        self.epoch_value[idx, 3] = smoothness
        self.epoch_acc[idx, 0] = train_acc
        self.epoch_acc[idx, 1] = test_acc
        self.current_epoch = idx + 1
        # return self.max_accuracy(False) == val_acc

    def max_accuracy(self, istrain):
        if self.current_epoch <= 0: return 0
        if istrain: return self.epoch_acc[:self.current_epoch, 0].max()
        else: return self.epoch_acc[:self.current_epoch, 1].max()

    def plot_curve(self, save_path):
        title = 'the train_loss/l_loss/u_loss/acc curve of train/val'
        dpi = 80
        width, height = 1200, 800
        legend_fontsize = 10
        scale_distance = 48.8
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 1
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_acc[:, 0]
        plt.plot(x_axis,
                 y_axis,
                 color='g',
                 linestyle='-',
                 label='train-accuracy',
                 lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_acc[:, 1]
        plt.plot(x_axis,
                 y_axis,
                 color='cyan',
                 linestyle='-',
                 label='valid-accuracy',
                 lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_value[:, 0]
        plt.plot(x_axis,
                 y_axis * 50,
                 color='g',
                 linestyle=':',
                 label='train-loss-x50',
                 lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_value[:, 1]
        plt.plot(x_axis,
                 y_axis * 50,
                 color='b',
                 linestyle=':',
                 label='l-loss-x50',
                 lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_value[:, 2]
        plt.plot(x_axis,
                 y_axis * 50,
                 color='r',
                 linestyle=':',
                 label='u-loss-x50',
                 lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)


        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print('---- save figure {} into {}'.format(title, save_path))

        plt.figure()
        y_axis[:] = self.epoch_value[:, 3]
        plt.plot(x_axis,
                 y_axis * 50,
                 color='y',
                 linestyle=':',
                 label='smoothness-x50',
                 lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print('---- save figure {} into {}'.format(title, save_path))

        plt.close(fig)


class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""

    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        assert total_epoch > 0
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2),
                                     dtype=np.float32)  # [epoch, train/val]
        self.epoch_losses = self.epoch_losses - 1

        self.epoch_accuracy = np.zeros((self.total_epoch, 2),
                                       dtype=np.float32)  # [epoch, train/val]
        self.epoch_accuracy = self.epoch_accuracy

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        assert idx >= 0 and idx < self.total_epoch, 'total_epoch : {} , but update with the {} index'.format(
            self.total_epoch, idx)
        self.epoch_losses[idx, 0] = train_loss
        self.epoch_losses[idx, 1] = val_loss
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1
        # return self.max_accuracy(False) == val_acc

    def max_accuracy(self, istrain):
        if self.current_epoch <= 0: return 0
        if istrain: return self.epoch_accuracy[:self.current_epoch, 0].max()
        else: return self.epoch_accuracy[:self.current_epoch, 1].max()

    def plot_curve(self, save_path):
        title = 'the accuracy/loss/consistency curve of train/val'
        dpi = 80
        width, height = 1200, 800
        legend_fontsize = 10
        scale_distance = 48.8
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 5
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis,
                 y_axis,
                 color='g',
                 linestyle='-',
                 label='train-accuracy',
                 lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis,
                 y_axis,
                 color='y',
                 linestyle='-',
                 label='valid-accuracy',
                 lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis,
                 y_axis * 50,
                 color='g',
                 linestyle=':',
                 label='train-loss-x50',
                 lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis,
                 y_axis * 50,
                 color='y',
                 linestyle=':',
                 label='valid-loss-x50',
                 lw=2)
        plt.legend(fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path.split('.')[0]+'_sm.pdf', dpi=dpi, bbox_inches='tight')
            print('---- save figure {} into {}'.format(title, save_path))
        plt.close(fig)


def time_string():
    ISOTIMEFORMAT = '%Y-%m-%d %X'
    string = '[{}]'.format(
        time.strftime(ISOTIMEFORMAT, time.gmtime(time.time())))
    return string


def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600 * need_hour) / 60)
    need_secs = int(epoch_time - 3600 * need_hour - 60 * need_mins)
    return need_hour, need_mins, need_secs


def time_file_str():
    ISOTIMEFORMAT = '%Y-%m-%d'
    string = '{}'.format(time.strftime(ISOTIMEFORMAT,
                                       time.gmtime(time.time())))
    return string + '-{}'.format(random.randint(1, 10000))
