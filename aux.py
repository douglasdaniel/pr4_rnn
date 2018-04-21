#!/usr/bin/env python3


import numpy as np
from math import sqrt


def distance_1_frame(pre, lab):
    # Takes 2 (Nx2) arrays and returns distance for each joint
    dist_1_frame = []
    for P,Q in zip(pre,lab):
        X = (Q[0] - P[0])**2
        Y = (Q[1] - P[1])**2
        dist = sqrt(X + Y)
        dist_1_frame.append(dist)
    return np.asarray(dist_1_frame)

def dist_1_sequence(seq_pred,seq_labl):
    # Takes 2 (NxMx2) arrays and distances for a sequence
    dist_1_seq = []
    for P,Q in zip(seq_pred,seq_labl):
        dist_frame = distance_1_frame(P,Q)
        dist_1_seq.append(dist_frame)
    return np.asarray(dist_1_seq)

def dist_1_batch(predict, label):
    # Tales a batch of predictions and labels and returns
    # the distance between them for each joint
    dist_1_batch = []
    for P,L in zip(predict,label):
        dist_sequence = dist_1_sequence(P,L)
        dist_1_batch.append(dist_sequence)
    return np.asarray(dist_1_batch)

def accuracy(predicts, labels):
    distances = dist_1_batch(predicts,labels)
    avg_error = np.mean(distances)
    avg_err_per_joint = np.mean(distances,axis=(0,1))
    return avg_error, avg_err_per_joint
