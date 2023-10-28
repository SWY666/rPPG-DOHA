# Copyright ©2022 Sun weiyu and Chen ying. All Rights Reserved.
from torch.utils.data import DataLoader, Dataset
import os
import torch
import cv2
import numpy as np
import random
# import visdom
from einops import rearrange
import pandas as pd
from scipy.signal import hilbert
import matplotlib.pyplot as plt
import math

def rect_search(rect_list, order):
    for i in rect_list:
        if i[0] == order:
            return i[1]


def Norm(wave):
    wave = np.array(wave)
    result = (wave - np.min(wave)) / (np.max(wave) - np.min(wave))
    return result


def fft2d(attn):
    attns = rearrange(attn, "b n t1 t2 -> (b n) t1 t2").unsqueeze(-1)
    imag_part = torch.zeros(attns.shape)
    complex = torch.cat([attns, imag_part], dim=-1)
    try:
        result = torch.fft(complex, 2)
    except:
        result = torch.fft.fft(complex, 2)
    result_list = torch.split(result, 1, dim=-1)
    result = torch.sqrt(torch.square(result_list[0]) + torch.square(result_list[1])).squeeze()
    result = rearrange(result, "(b n) t1 t2 -> b n t1 t2", n=8)
    return result


def diff_distance(input):
    result_list = []
    for i in range(input.shape[-2]):
        result_list.append(torch.cat([torch.abs(input[i, :] - input[j, :]) for j in range(input.shape[-2])]).unsqueeze(-2))

    result = torch.cat(result_list, -2)
    return result


def self_similarity_calc(ippg):
    ippg_phase0 = myhilbert(ippg)
    ippg_phase = amass_hilbort(ippg_phase0)[1:]
    result_list = []
    for i in range(len(ippg_phase)):
        tmp_list = []
        for j in range(len(ippg_phase)):
            similarity = np.cos(ippg_phase[i] - ippg_phase[j])
            tmp_list.append(similarity)
        tmp_list = torch.FloatTensor(tmp_list).unsqueeze(-1)
        result_list.append(tmp_list)
    result = torch.cat(result_list, dim=-1)
    return result


def amass_hilbort(ippg):
    peak_record = [(0, 0)]
    current_sum = 0
    for i in range(1, len(ippg)):
        if (ippg[i] - ippg[i - 1]) < 0:
            current_sum += ippg[i - 1] - ippg[i]
            peak_record.append((i, current_sum))

    sum_list = [peak_record[i][1] for i in range(len(peak_record))]
    peak_record.append((len(ippg), None))
    record_list = [(peak_record[i][0], peak_record[i + 1][0]) for i in range(len(peak_record) - 1)]
    result =[]
    for i in range(len(ippg)):
        for j in range(len(record_list)):
            if record_list[j][0] <= i < record_list[j][1]:
                ans = ippg[i] + sum_list[j]
                while ans > 2 * np.pi:
                    ans -= 2 * np.pi
                result.append(ans)

    return result


def cal_polarity(ippg):
    # start_polarity = 1 if (ippg[1] - ippg[0]) > 0 else -1
    total_polarity = []
    for i in range(1, len(ippg)):
        total_polarity.append(1 if (ippg[i] - ippg[i - 1]) > 0 else -1)
    return total_polarity


def myhilbert(ippg_test):
    ippg_hilbert = hilbert(ippg_test)
    N = len(ippg_test)
    ippg_hilbert_phase = np.zeros(N)
    ippg_hilbert_phase_shift = np.zeros(N)
    for i in range(N):
        if ippg_hilbert[i].real == 0:
            if (ippg_hilbert[i].imag > 0):
                ippg_hilbert_phase[i] = 1.57
            elif (ippg_hilbert[i].imag < 0):
                ippg_hilbert_phase[i] = -1.57
            else:
                ippg_hilbert_phase[i] = 0
        else:
            ippg_hilbert_phase[i] = math.atan(ippg_hilbert[i].imag / ippg_hilbert[i].real)
    k = 1
    for i in range(N):
        if i != 0 and ippg_hilbert_phase[i] - ippg_hilbert_phase[i - 1] < -1.5:
            k = -k
        ippg_hilbert_phase_shift[i] = k * ippg_hilbert_phase[i]
    return ippg_hilbert_phase


if __name__ == "__main__":
    pass

