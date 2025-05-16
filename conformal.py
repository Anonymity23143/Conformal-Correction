import numpy as np
import torch
import math

device = torch.device('cuda:0')

def calculate_entropy(probabilities):
    entropy_list = []
    for i in range(np.shape(probabilities)[0]):
        pro = probabilities[i]
        entropy = 0.0
        for p in pro:
            if p > 0:
                entropy -= p * math.log(p, 2)
        entropy_list.append(entropy)
    return np.array(entropy_list)

def tps(cal_smx, val_smx, cal_labels, val_labels, n, alpha):
    cal_scores = 1 - cal_smx[np.arange(n), cal_labels]
    q_level = np.ceil((n+1) * (1-alpha)) / n
    qhat = np.quantile(cal_scores, q_level, method='higher')
    # print('qhat: ', qhat)
    prediction_sets = val_smx >= (1-qhat)
    cov = prediction_sets[np.arange(prediction_sets.shape[0]),val_labels].mean()
    eff = np.sum(prediction_sets) / len(prediction_sets)
    
    num_classes = prediction_sets.shape[1]  
    coverage_per_class = np.zeros(num_classes)

    new_labels = cal_scores <= qhat
    new_labels = new_labels.astype(int)
    
    for cls_id in range(num_classes):
        class_mask = (val_labels == cls_id)  
        class_predictions = prediction_sets[class_mask]  
        class_labels = val_labels[class_mask]  
        
        if len(class_labels) > 0:
            class_coverage = class_predictions[np.arange(len(class_labels)), class_labels].mean()
            coverage_per_class[cls_id] = class_coverage
        else:
            coverage_per_class[cls_id] = np.nan
    
    return coverage_per_class, prediction_sets, cov, eff, new_labels

def aps(cal_smx, val_smx, cal_labels, val_labels, n, alpha):
    cal_pi = cal_smx.argsort(1)[:, ::-1]
    cal_srt = np.take_along_axis(cal_smx, cal_pi, axis=1).cumsum(axis=1)
    cal_scores = np.take_along_axis(cal_srt, cal_pi.argsort(axis=1), axis=1)[
        range(n), cal_labels
    ]
    qhat = np.quantile(
        cal_scores, np.ceil((n + 1) * (1 - alpha)) / n, method="higher"
    )
    val_pi = val_smx.argsort(1)[:, ::-1]
    val_srt = np.take_along_axis(val_smx, val_pi, axis=1).cumsum(axis=1)
    prediction_sets = np.take_along_axis(val_srt <= qhat, val_pi.argsort(axis=1), axis=1)
    cov = prediction_sets[np.arange(prediction_sets.shape[0]),val_labels].mean()
    eff = np.sum(prediction_sets)/len(prediction_sets)

    cal_c = np.where(cal_scores <= qhat, 1.0, 0.0)
    e = calculate_entropy(cal_smx)
    pos = (cal_c * e).sum() / cal_c.sum()
    neg = ((1 - cal_c) * e).sum() / (1 - cal_c).sum()
    
    cal_prediction_sets = np.take_along_axis(cal_srt <= qhat, cal_pi.argsort(axis=1), axis=1)
    cal_c_matrix = np.tile(np.reshape(cal_c, (-1, 1)), np.shape(cal_prediction_sets)[1])
    pos_eff = np.sum(cal_c_matrix * cal_prediction_sets) / np.sum(cal_c)
    neg_eff = np.sum((1 - cal_c_matrix) * cal_prediction_sets) / np.sum(1 - cal_c)
    
    return prediction_sets, cov, eff, pos, neg, pos_eff, neg_eff, qhat



def raps(cal_smx, val_smx, cal_labels, val_labels, n, alpha):
    # lam_reg = 0.01
    lam_reg = 0.1
    k_reg = min(5, cal_smx.shape[1])
    k_reg = 5
    disallow_zero_sets = False
    rand = True
    reg_vec = np.array(k_reg*[0,] + (cal_smx.shape[1]-k_reg)*[lam_reg,])[None,:]

    cal_pi = cal_smx.argsort(1)[:,::-1]; 
    cal_srt = np.take_along_axis(cal_smx,cal_pi,axis=1)
    cal_srt_reg = cal_srt + reg_vec
    cal_L = np.where(cal_pi == cal_labels[:,None])[1]
    cal_scores = cal_srt_reg.cumsum(axis=1)[np.arange(n),cal_L] - np.random.rand(n)*cal_srt_reg[np.arange(n),cal_L]
    # cal_scores = cal_srt_reg.cumsum(axis=1)[np.arange(n),cal_L] #- cal_srt_reg[np.arange(n),cal_L]
    # Get the score quantile
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, method='higher')
    # Deploy
    n_val = val_smx.shape[0]
    val_pi = val_smx.argsort(1)[:,::-1]
    val_srt = np.take_along_axis(val_smx,val_pi,axis=1)
    val_srt_reg = val_srt + reg_vec
    val_srt_reg_cumsum = val_srt_reg.cumsum(axis=1)
    indicators = (val_srt_reg.cumsum(axis=1) - np.random.rand(n_val,1)*val_srt_reg) <= qhat if rand else val_srt_reg.cumsum(axis=1) - val_srt_reg <= qhat
    # indicators = val_srt_reg.cumsum(axis=1) - val_srt_reg <= qhat
    # indicators = val_srt_reg.cumsum(axis=1) <= qhat
    if disallow_zero_sets: indicators[:,0] = True
    prediction_sets = np.take_along_axis(indicators,val_pi.argsort(axis=1),axis=1)
    cov = prediction_sets[np.arange(prediction_sets.shape[0]),val_labels].mean()
    eff = np.sum(prediction_sets)/len(prediction_sets)

    cal_c = np.where(cal_scores <= qhat, 1.0, 0.0)
    e = calculate_entropy(cal_smx)
    pos = (cal_c * e).sum() / cal_c.sum()
    neg = ((1 - cal_c) * e).sum() / (1 - cal_c).sum()
    
    # cal_prediction_sets = np.take_along_axis(cal_srt <= qhat, cal_pi.argsort(axis=1), axis=1)
    # cal_c_matrix = np.tile(np.reshape(cal_c, (-1, 1)), np.shape(cal_prediction_sets)[1])
    # pos_eff = np.sum(cal_c_matrix * cal_prediction_sets) / np.sum(cal_c)
    # neg_eff = np.sum((1 - cal_c_matrix) * cal_prediction_sets) / np.sum(1 - cal_c)

    
    return prediction_sets, cov, eff, pos, neg, 0.0, 0.0, qhat


def saps(cal_smx, val_smx, cal_labels, val_labels, n, alpha, lambda_weight=0.01):
    # u = np.random.uniform(0, 1)
    u = 0.5
    cal_pi = cal_smx.argsort(1)[:, ::-1]  
    cal_srt = np.take_along_axis(cal_smx, cal_pi, axis=1).cumsum(axis=1) 
    
    cal_scores = np.zeros_like(cal_labels, dtype=float)
    
    for i in range(n):
        label_non_conformity_scores = np.zeros(cal_smx.shape[1], dtype=float)
        
        max_prob_label = cal_pi[i, 0]
        label_non_conformity_scores[max_prob_label] = cal_smx[i, max_prob_label] * u
        
        for rank in range(1, cal_smx.shape[1]):
            label = cal_pi[i, rank]
            non_conformity_score = cal_smx[i, max_prob_label] + (rank - 1 + u) * lambda_weight
            label_non_conformity_scores[label] = non_conformity_score
        
        correct_label = cal_labels[i]
        cal_scores[i] = label_non_conformity_scores[correct_label]

    qhat = np.quantile(
        cal_scores, np.ceil((n + 1) * (1 - alpha)) / n, method="higher"
    )
    
    val_pi = val_smx.argsort(1)[:, ::-1]
    val_srt = np.take_along_axis(val_smx, val_pi, axis=1).cumsum(axis=1)

    prediction_sets = np.zeros_like(val_srt, dtype=bool)

    for i in range(val_smx.shape[0]):
        max_prob_label = val_pi[i, 0]
        prediction_sets[i, max_prob_label] = ((val_smx[i, max_prob_label] * u) <= qhat)
        
        for rank in range(1, val_smx.shape[1]):
            label = val_pi[i, rank]
            non_conformity_score = val_smx[i, max_prob_label] + (rank - 1 + u) * lambda_weight
            prediction_sets[i, label] = (non_conformity_score <= qhat)
    
    cov = prediction_sets[np.arange(prediction_sets.shape[0]), val_labels].mean()
    eff = np.sum(prediction_sets) / len(prediction_sets)

    cal_c = np.where(cal_scores <= qhat, 1.0, 0.0)
    e = calculate_entropy(cal_smx)
    pos = (cal_c * e).sum() / cal_c.sum()
    neg = ((1 - cal_c) * e).sum() / (1 - cal_c).sum()
    
    # cal_prediction_sets = np.take_along_axis(cal_srt <= qhat, cal_pi.argsort(axis=1), axis=1)
    # cal_c_matrix = np.tile(np.reshape(cal_c, (-1, 1)), np.shape(cal_prediction_sets)[1])
    # pos_eff = np.sum(cal_c_matrix * cal_prediction_sets) / np.sum(cal_c)
    # neg_eff = np.sum((1 - cal_c_matrix) * cal_prediction_sets) / np.sum(1 - cal_c)
    
    return prediction_sets, cov, eff, pos, neg, 0.0, 0.0, qhat

