# Copyright ©2022 Sun weiyu and Chen ying. All Rights Reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

class Frequency_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, target, ground_truth):
        frequency_blocks_gt = torch.matmul(ground_truth.unsqueeze(-1), ground_truth.unsqueeze(1))
        frequency_blocks_tg = torch.matmul(target, target.transpose(-1, -2))
        comparison = frequency_blocks_gt.mul(frequency_blocks_tg)
        comparison = rearrange(comparison, "B H W -> B (H W)").mean(dim=1).mean()
        return comparison


class Mask_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, masks, real_masks):
        loss_list = []
        for i in range(masks.shape[0]):
            loss_list.append(self.loss_fn(masks[i], real_masks[i]).unsqueeze(0))
        final_sum = torch.cat(loss_list).mean()
        return final_sum


class Batch_MSE_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, pred, target):
        tmp = pred - target
        tmp = torch.mul(tmp, tmp)
        tmp = torch.mean(tmp, dim=-1)
        tmp = torch.mean(tmp, dim=-1)
        return tmp


def nagative_pearson_correlation(x1, y1, beta=1e-7):
    # return 1 - ((x1*y1).sum()*x1.shape[0] - x1.sum()*y1.sum())/(torch.sqrt(x1.shape[0]*(x1*x1).sum() - (x1.sum())**2 + beta)*torch.sqrt(y1.shape[0] * (y1 * y1).sum() - (y1.sum()) ** 2 + beta))
    return 1 - ((x1 - x1.mean())*(y1 - y1.mean())).sum() / (torch.sqrt(((x1 - x1.mean())**2).sum() + beta) * torch.sqrt(((y1 - y1.mean())**2).sum() + beta))


class batch_pearson_loss_wave(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predict, target):
        P_value = []
        for i in range(predict.shape[0]):
            P_value.append(nagative_pearson_correlation(predict[i, :], target[i, :]).unsqueeze(0))
        result = torch.cat(P_value).mean()
        return result


class N_Pearson_Correlation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predict, target):
        # predicts = predict.split([1 for i in range(predict.shape[0])], dim=0)
        # P_value = []
        # # print(len(predict))
        # for i in range(len(predicts)):
        #     P_value.append(nagative_pearson_correlation(predict[i, :], target[i, :]).unsqueeze(0))
        # result = torch.cat(P_value).mean()
        # return result
        return pearson_matrix(predict, target)


def pearson_matrix(x, y, batch_first=True, beta=1e-7):
    assert x.shape == y.shape

    if batch_first:
        dim = -1
    else:
        dim = 0

    centered_x = x - torch.mean(x, dim=dim, keepdim=True)
    centered_y = y - torch.mean(y, dim=dim, keepdim=True)

    covariance = torch.sum(centered_x * centered_y, dim=dim, keepdim=True)

    bessel_corrected_covariance = covariance / (x.shape[dim] - 1)

    x_std = torch.std(x, dim=dim, keepdim=True)
    y_std = torch.std(y, dim=dim, keepdim=True)

    corr = 1 - bessel_corrected_covariance / (x_std * y_std + beta)
    return corr.mean()


class ATTN_LOSS(nn.Module):
    def __init__(self):
        super().__init__()
        self.attend = nn.Softmax(dim=-1)
        self.attend1 = nn.Softmax(dim=-2)
        self.Loss_fn = nn.MSELoss()

    def forward(self, predict, target):
        #target -> (batch_size, len_of_ticks)
        # attn_target = torch.matmul(target.unsqueeze(-1), target.unsqueeze(-2)) # (batch_size, len_of_ticks, len_of_ticks)
        # attn_target_softmax_version = (self.attend(attn_target) + self.attend1(attn_target)) / 2
        error = self.Loss_fn(predict, target)
        return error


# class attn Pearson
class ATTN_LOSS_Pearson(nn.Module):
    def __init__(self):
        super().__init__()
        self.attend = nn.Softmax(dim=-1)
        self.attend1 = nn.Softmax(dim=-2)
        self.Loss_fn = nn.MSELoss()
        self.Pearson_Loss = N_Pearson_Correlation()

    def forward(self, predict, target):
        result = 0
        for i in range(predict.shape[0]):
            losses = self.Pearson_Loss(predict[i], target[i])
            result += losses
        error = result / predict.shape[0]
        return error





if __name__ == "__main__":
    input1 = torch.FloatTensor(np.random.random((3, 50, 50)))
    ground_truth = torch.FloatTensor(np.random.random((3, 50)))
    model = ATTN_LOSS()
    print(model(input1, ground_truth))