import numpy as np
from copy import copy as copy
from copy import deepcopy as deepcopy
import pdb
from scipy import stats
import numpy as np
from sklearn.metrics import pairwise_distances
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
"""
This is an implementation of baseline method -- Badge
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_grad_embedding(model2test, test_set, n_classes=10, embDim = 1024):
    """ gradient embedding (assumes cross-entropy loss) of the last layer"""
    model2test.eval()
    print(embDim,n_classes)
    nLab = n_classes

    embedding = np.zeros([len(test_set), embDim * nLab])
    loader_te = torch.utils.data.DataLoader(test_set,  
                                                pin_memory=True, shuffle=False)
    idx = 0
    with torch.no_grad():
        for x, y in loader_te:
            x, y = x.to(device), y.to(device) 
            cout, out = model2test(x)
            out = out.data.cpu().numpy()
            batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
            maxInds = np.argmax(batchProbs,1)
            for j in range(len(y)):
                for c in range(nLab):
                    if c == maxInds[j]:
                        embedding[idx][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (1 - batchProbs[j][c])
                    else:
                        # print('idx:',idx,' c:',c,'j:', j)
                        embedding[idx][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (-1 * batchProbs[j][c])
                idx += 1
        return torch.Tensor(embedding)
        
def init_centers(X, K):
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
    print('#Samps\tTotal Distance')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] >  newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
        if sum(D2) == 0.0: pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2)/ sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    gram = np.matmul(X[indsAll], X[indsAll].T)
    val, _ = np.linalg.eig(gram)
    val = np.abs(val)
    vgt = val[val > 1e-2]
    return indsAll


def badge_selection(model2test, test_set, budget, embDim):
    grad_embedding = get_grad_embedding(model2test, test_set, embDim = embDim)
    idxs_unlabeled = np.arange(np.shape(grad_embedding)[0])

    chosen = init_centers(grad_embedding.numpy(), budget)
    return list(idxs_unlabeled[chosen])

