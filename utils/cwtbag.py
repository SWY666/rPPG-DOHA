# Copyright ©2022 Sun weiyu and Chen ying. All Rights Reserved.
import numpy as np
import math
import pycwt

def maxscale(array1):
    list1 = []
    for i in range(array1.shape[0]):
        list1.append(np.mean(array1[i]))
    return np.array(list1)


def rowcal(array1, row1):
    array2 = np.zeros(array1.shape)
    for i in range(array1.shape[1]):
        array2[:,i] = array1[:, i] * row1
    return array2


frequencies = [3.75, 3.720703125, 3.69140625, 3.662109375, 3.6328125, 3.603515625, 3.5742187500000004, 3.544921875,
               3.515625, 3.486328125, 3.45703125, 3.427734375, 3.3984375, 3.369140625, 3.33984375, 3.310546875,
               3.28125,
               3.251953125, 3.22265625, 3.193359375, 3.1640625, 3.134765625, 3.10546875, 3.076171875, 3.046875,
               3.017578125,
               2.98828125, 2.958984375, 2.9296875, 2.9003906249999996, 2.87109375, 2.841796875, 2.8125, 2.783203125,
               2.75390625, 2.7246093750000004, 2.6953125, 2.666015625, 2.6367187500000004, 2.607421875, 2.578125,
               2.548828125, 2.51953125, 2.490234375, 2.4609375, 2.431640625, 2.40234375, 2.373046875, 2.34375,
               2.314453125,
               2.28515625, 2.255859375, 2.2265625, 2.197265625, 2.16796875, 2.138671875, 2.109375, 2.080078125,
               2.05078125,
               2.021484375, 1.9921875, 1.962890625, 1.93359375, 1.904296875, 1.875, 1.845703125, 1.81640625,
               1.7871093750000002, 1.7578125, 1.728515625, 1.69921875, 1.669921875, 1.640625, 1.611328125,
               1.58203125,
               1.552734375, 1.5234375, 1.494140625, 1.46484375, 1.435546875, 1.40625, 1.376953125, 1.34765625,
               1.3183593750000002, 1.2890625, 1.259765625, 1.23046875, 1.201171875, 1.171875, 1.142578125,
               1.11328125,
               1.083984375, 1.0546875, 1.025390625, 0.99609375, 0.966796875, 0.9375, 0.908203125, 0.87890625,
               0.849609375,
               0.8203125, 0.791015625, 0.76171875, 0.732421875, 0.703125, 0.673828125, 0.64453125]


def cwt_filtering(listin, samplingrate, frequencies=frequencies):
    sr = samplingrate
    plf1 = np.array(listin)
    result = pycwt.cwt(plf1, 1 / sr, freqs=np.array(frequencies))
    cwtmatr = result[0]
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.imshow(abs(result[0]))
    # plt.show()
    scale1 = maxscale(abs(result[0]))
    co = np.argmax(scale1)
    myguasswindow = np.array([0.0 for x in range(len(scale1))])
    for j in range(len(scale1)):
        # myguasswindow[j] = math.exp(-0.2 * ((j - co) / (0.08 * len(scale1))) ** 2)  # ×, for 3D CNN
        # myguasswindow[j] = math.exp(-0.35 * ((j - co) / (0.08 * len(scale1))) ** 2)  # ×, for CAN
        myguasswindow[j] = math.exp(-0.5 * ((j - co) / (0.08 * len(scale1))) ** 2)  # √
    mycwtmatr = rowcal(abs(result[0]), myguasswindow)
    mycwtmatr2 = rowcal(result[0].real, myguasswindow)
    result_copy = result[1][:]
    result3 = pycwt.icwt(mycwtmatr2, result_copy, 1 / sr).real
    return result3, mycwtmatr, cwtmatr


def cwt_show(listin, sr):
    result = pycwt.cwt(listin, 1 / sr, freqs=np.array(frequencies))
    cwtmatr = abs(result[0])
    return cwtmatr


if __name__ == "__main__":
    pass