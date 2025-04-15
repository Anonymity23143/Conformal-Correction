import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from scipy.special import lambertw
import numpy as np

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma = 0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        logpt = F.log_softmax(inputs)
        targets = targets.view(-1, 1)
        logpt = logpt.gather(1, targets)
        logpt = logpt.view(-1)
        pt = logpt.exp()
        loss = -1 * (1-pt)**self.gamma * logpt

        return loss.mean()











# def get_gamma(p = 0.2):
#     '''
#     Get the gamma for a given pt where the function g(p, gamma) = 1
#     '''
#     y = ((1-p)**(1-(1-p)/(p*np.log(p)))/(p*np.log(p)))*np.log(1-p)
#     gamma_complex = (1-p)/(p*np.log(p)) + lambertw(-y + 1e-12, k=-1)/np.log(1-p)
#     gamma = np.real(gamma_complex) # gamma for which p_t > p results in g(p_t,gamma)<1
#     return gamma

# ps = [0.2, 0.5]
# gammas = [5.0, 2.0]
# i = 0
# gamma_dic = {}
# for p in ps:
#     gamma_dic[p] = gammas[i]
#     i += 1

# class FocalLossAdaptive(nn.Module):
#     def __init__(self, gamma = 0, device = None):
#         super(FocalLossAdaptive, self).__init__()
#         self.gamma = gamma
#         self.device = device

#     def get_gamma_list(self, pt):
#         gamma_list = []
#         batch_size = pt.shape[0]
#         for i in range(batch_size):
#             pt_sample = pt[i].item()
#             if (pt_sample >= 0.5):
#                 gamma_list.append(self.gamma)
#                 continue
#             # Choosing the gamma for the sample
#             for key in sorted(gamma_dic.keys()):
#                 if pt_sample < key:
#                     gamma_list.append(gamma_dic[key])
#                     break
#         return torch.tensor(gamma_list).to(self.device)

#     def forward(self, inputs, targets):
#         logpt = F.log_softmax(inputs)
#         targets = targets.view(-1, 1)
#         logpt = logpt.gather(1, targets)
#         logpt = logpt.view(-1)
#         pt = logpt.exp()
#         gamma = self.get_gamma_list(pt)
#         loss = -1 * (1-pt)**gamma * logpt

#         return loss.mean()

class FocalLossAdaptive(nn.Module):
    def __init__(self, gamma = 0):
        super(FocalLossAdaptive, self).__init__()
        self.gamma = gamma

    # def get_gamma_list(self, pt):
    #     gamma_list = []
    #     batch_size = pt.shape[0]
    #     for i in range(batch_size):
    #         pt_sample = pt[i].item()
    #         if (pt_sample >= 0.5):
    #             gamma_list.append(self.gamma)
    #             continue
    #         # Choosing the gamma for the sample
    #         for key in sorted(gamma_dic.keys()):
    #             if pt_sample < key:
    #                 gamma_list.append(gamma_dic[key])
    #                 break
    #     return torch.tensor(gamma_list).to(self.device)

    def forward(self, inputs, targets):
        logpt = F.log_softmax(inputs)
        targets = targets.view(-1, 1)
        logpt = logpt.gather(1, targets)
        logpt = logpt.view(-1)
        pt = logpt.exp()
        # gamma = self.get_gamma_list(pt)
        gamma = 10.0 * pt
        loss = -1 * (1-pt)**gamma * logpt

        return loss.mean()
    




class EntropyLoss(torch.nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, inputs):
        pt = inputs
        # print(torch.sum(pt, dim=1))
        logpt = torch.log2(pt)
        loss = torch.sum(-pt * logpt, dim=1)
        return loss