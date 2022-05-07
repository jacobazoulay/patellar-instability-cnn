import numpy as np
import torch
#from pykeops.torch import LazyTensor


def getconfmatrix(args, label_count):
    confusion_matrix = np.zeros((args.num_classes, args.num_classes))
    for i_label in range(args.num_classes):
        for i_pred in range(args.num_classes):
            cur_index = i_label * args.num_classes + i_pred
            confusion_matrix[i_label, i_pred] = label_count[cur_index]
    return confusion_matrix


def getLabelCount(args, pred, gt, ignore=-1):
    size = gt.shape
    C = args.num_classes
    label_count = np.zeros((C*C + C))
    seg_gt = np.asarray(gt[:, :size[-2], :size[-1]], dtype=np.int)
    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    pred = pred[ignore_index]
    index = (seg_gt * args.num_classes + pred).astype('int32')
    bins = np.bincount(index)
    label_count[0:len(bins)] = bins
    return label_count


def knn(args, x_train, x_test, K):
    S = x_test.shape
    x_test = x_test.view(-1, S[-1])
    y_train = torch.tensor(np.array(range(args.num_classes))).cuda()
    X_i = LazyTensor(x_test[:, None, :])  # (10000, 1, 784) test set
    X_j = LazyTensor(x_train[None, :, :])  # (1, 60000, 784) train set
    D_ij = ((X_i - X_j) ** 2).sum(-1)  # (10000, 60000) symbolic matrix of squared L2 distances

    ind_knn = D_ij.argKmin(K, dim=1)  # Samples <-> Dataset, (N_test, K)
    lab_knn = y_train[ind_knn]  # (N_test, K) array of integers in [0,9]
    y_knn, _ = lab_knn.mode()   # Compute the most likely label
    torch.cuda.synchronize()
    return y_knn


def getprediction(args, neigh, code, labels):
    N, w, h = labels.shape
    code = code.unsqueeze(0)
    idx1 = knn(args, args.matembd, code, K=1)
    pred = idx1.view(N, w, h)
    return pred.detach().cpu().numpy()


def getmeaniou(args, confusion_matrix):
    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    IoU = []
    for i in range(args.num_classes):
        if pos[i] > 0:
            IoU_i = tp[i] / np.maximum(1.0, res[i] + pos[i] - tp[i])
            IoU.append(IoU_i)
    meaniou = np.mean(IoU)
    return meaniou, res, pos, tp