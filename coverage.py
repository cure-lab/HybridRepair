from sklearn.metrics import pairwise_distances
import numpy as np
import torch
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def coreset_coverage(X, lb_idxes, log):
#     """
#         X: The whole dataset
#         ib_idxes (Boolean array): The indexes of the selected labeled data
#         output: max/average radius which cover the 100%/98% of the dataset
#     """
#     # print (lb_idxes)
#     spare_room = int(0.002*len(X))
#     print_log("spare room: {}".format(spare_room), log)
#     embedding = X

#     ub_idxes = list(set(range(len(X))) - set(lb_idxes))
#     dist_ctr = pairwise_distances(embedding[ub_idxes], embedding[lb_idxes])
#     # group unlabeled data to their nearest labeled data
#     min_args = np.argmin(dist_ctr, axis=1)
#     print_log("min args: {}".format(min_args), log)
#     delta = []
#     gains = []
#     group_sizes = []
#     for j in np.arange(len(lb_idxes)):
#         # get the sample index for the jth center
#         idxes = np.nonzero(min_args == j)[0]
#         distances = dist_ctr[idxes, j]
#         group_size = len(distances)
        
#         # print_log('group size: {}'.format(group_size), log)

#         delta_j = 0 if len(distances)==0 else distances.max()
#         # how much delta reduction can be reached by using the spare room
#         if group_size >= spare_room:
#             gain_by_spare = distances.max() - np.sort(distances)[::-1][spare_room-1]
#         else:
#             gain_by_spare = distances.max() if len(distances) != 0 else 0
#         # print_log("Gain by spare room: {}".format(gain_by_spare), log)
#         delta.append(delta_j)
#         gains.append(gain_by_spare)
#         group_sizes.append(group_size)

#     # full cover
#     coverage_mean = np.array(delta).mean()
#     coverage_max = np.array(delta).max()
#     coverage_topmean = np.sort(delta)[::-1][:int(len(delta)*0.3)].mean()

#     # relax cover
#     # pick the group which has most gains
#     group_idx = np.argmax(gains)
#     # revise the corresponding delta
#     delta[group_idx] -= gains[group_idx]

#     coverage_mean98 = np.array(delta).mean()

#     return coverage_max, coverage_mean, coverage_mean98, coverage_topmean 


def coreset_coverage(X, lb_idxes, log):
    """
        X: The whole dataset
        ib_idxes (Boolean array): The indexes of the selected labeled data
        output: max/average radius which cover the 100%/98% of the dataset
    """
    embedding = X

    ub_idxes = list(set(range(len(X))) - set(lb_idxes))
    dist_ctr = pairwise_distances(embedding[ub_idxes], embedding[lb_idxes])
    # group unlabeled data to their nearest labeled data
    min_args = np.argmin(dist_ctr, axis=1)
    print_log("min args: {}".format(min_args), log)
    delta = []
    for j in np.arange(len(lb_idxes)):
        # get the sample index for the jth center
        idxes = np.nonzero(min_args == j)[0]
        distances = dist_ctr[idxes, j]
        
        delta_j = 0 if len(distances)==0 else distances.max()
        delta.append(delta_j)

    # full cover
    coverage_mean = np.array(delta).mean()
    coverage_max = np.array(delta).max()
    coverage_topmean = np.sort(delta)[::-1][:int(len(delta)*0.3)].mean()

    return coverage_max, coverage_mean, -1, coverage_topmean 


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()
