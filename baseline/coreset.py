import numpy as np
from sklearn.metrics import pairwise_distances

def coreset_selection(test_latents, budget):
    X = test_latents
    idxs_unlabeled = np.arange(np.shape(X)[0])
    m = np.shape(X)[0]
    min_dist = np.tile(float("inf"), m)
 
    idxs = []

    for i in range(budget):
        idx = min_dist.argmax()
        idxs.append(idx)
        dist_new_ctr = pairwise_distances(X, X[[idx], :])
        for j in range(m):
            min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])


    return list(idxs_unlabeled[idxs])

