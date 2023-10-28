# Copyright ©2022 Sun weiyu and Chen ying. All Rights Reserved.
import numpy as np
from utils.cwtbag import cwt_filtering
from scipy import signal
from scipy.signal import find_peaks, detrend
# import neurokit2 as nk
from scipy.fft import fft

image_size = 131
black_size = 64
dif = 12

def peakcheckez(a, samplingrate):
    result = []
    for i in range(len(a)):
        if i == 0 or i == len(a) - 1:
            pass
        else:
            if a[i] >= a[i - 1] and a[i] > a[i + 1]:
                result.append(i)

    hr_list = []
    if len(result) <= 1:  # during training, we only expect 2 peaks at least.
        hr = 0
    else:
        for i in range(len(result) - 1):
            hr = 60 * samplingrate / (result[i + 1] - result[i])
            hr_list.append(hr)
        hr = np.mean(np.array(hr_list))
    return hr


def my_peak_detect(sig, samplingrate):
    bvp_detrend = detrend(sig, type='linear')  # useless...
    # bvp_detrend = polynomial(bvp, order=10)
    # # visualize detrend.
    # plt.figure()
    # plt.plot(bvp)
    # plt.plot(bvp_detrend)

    height = np.mean(bvp_detrend)  # minimal required height.
    distance = samplingrate / 2  # Required minimal horizontal distance (>= 1) in samples between neighbouring peaks.
    peaks = find_peaks(bvp_detrend, height=height, distance=distance)[0]

    hr_list = []
    if len(peaks) <= 1:  # during training, we only expect 2 peaks at least.
        hr = 0
    else:
        for i in range(len(peaks) - 1):
            hr = 60 * samplingrate / (peaks[i + 1] - peaks[i])
            hr_list.append(hr)
        hr = np.mean(np.array(hr_list))
    return hr
    # return hr, peaks



# def my_peak_detect(sig, samplingrate):
#     bvp_detrend = detrend(sig, type='linear')  # useless...
#     # bvp_detrend = polynomial(bvp, order=10)
#     # # visualize detrend.
#     # plt.figure()
#     # plt.plot(bvp)
#     # plt.plot(bvp_detrend)
#
#     height = np.mean(bvp_detrend)  # minimal required height.
#     distance = samplingrate / 3  # Required minimal horizontal distance (>= 1) in samples between neighbouring peaks.
#     peaks = find_peaks(bvp_detrend, height=height, distance=distance)[0]
#
#     hr_list = []
#     if len(peaks) <= 1:  # during training, we only expect 2 peaks at least.
#         hr = 0
#     else:
#         for i in range(len(peaks) - 1):
#             hr = 60 * samplingrate / (peaks[i + 1] - peaks[i])
#             hr_list.append(hr)
#         hr = np.mean(np.array(hr_list))
#     return hr, peaks


def hr_cal(tmp, samplingrate=30):
    # tmp = tmp[10:-10]
    # tmp = tmp[5:-5]  # shorter than testing.
    f1 = 0.8
    f2 = 3
    samplingrate = samplingrate
    b, a = signal.butter(4, [2 * f1 / samplingrate, 2 * f2 / samplingrate], 'bandpass')
    tmp = signal.filtfilt(b, a, np.array(tmp))
    tmp = cwt_filtering(tmp, samplingrate)[0]

    hr_caled = peakcheckez(tmp, samplingrate)
    return hr_caled, tmp


def my_hr_cal(tmp, samplingrate=30):
    # tmp = tmp[10:-10]
    # tmp = tmp[5:-5]  # shorter than testing.
    # f2 = 4
    # b, a = signal.butter(4, [2 * f2 / samplingrate], 'lowpass')
    # tmp = signal.filtfilt(b, a, np.array(tmp))

    f1 = 0.8
    f2 = 2.8
    samplingrate = samplingrate
    b, a = signal.butter(4, [2 * f1 / samplingrate, 2 * f2 / samplingrate], 'bandpass')
    tmp = signal.filtfilt(b, a, np.array(tmp))
    # tmp = cwt_filtering(tmp, samplingrate)[0]

    hr_caled, peaks = my_peak_detect(tmp, samplingrate)
    return hr_caled, tmp, peaks


def my_hr_cal_nk(ppg, samplingrate=30):
    hrv = {}
    sig_output, info_output = nk.ppg_process(ppg, sampling_rate=samplingrate)
    peaks = info_output['PPG_Peaks']
    ppg_output = nk.ppg_intervalrelated(sig_output, sampling_rate=samplingrate)
    hr = ppg_output['PPG_Rate_Mean'].to_list()[0]
    hrv['HRV_LFn'] = ppg_output['HRV_LFn'].to_list()[0]
    hrv['HRV_HFn'] = ppg_output['HRV_HFn'].to_list()[0]
    hrv['HRV_LFHF'] = ppg_output['HRV_LFHF'].to_list()[0]
    return hr, hrv, peaks


def hr_cal_freq(sig, samplingrate, harmonics_removal=True, filter=True):
    # get heart rate by FFT
    # return both heart rate and PSD
    if filter:
        f1 = 0.65
        f2 = 4
        b, a = signal.butter(4, [2 * f1 / samplingrate, 2 * f2 / samplingrate], 'bandpass')
        sig = signal.filtfilt(b, a, np.array(sig))
        sig = cwt_filtering(sig, samplingrate)[0]
    sig = sig.reshape(-1)
    sig = sig * signal.windows.hann(sig.shape[0])
    sig_f = np.abs(fft(sig))
    low_idx = np.round(0.6 / samplingrate * sig.shape[0]).astype('int')
    high_idx = np.round(4 / samplingrate * sig.shape[0]).astype('int')
    psd = sig_f.copy()

    sig_f[:low_idx] = 0
    sig_f[high_idx:] = 0

    peak_idx, _ = signal.find_peaks(sig_f)
    sort_idx = np.argsort(sig_f[peak_idx])
    sort_idx = sort_idx[::-1]

    peak_idx1 = peak_idx[sort_idx[0]]
    peak_idx2 = peak_idx[sort_idx[1]]

    f_hr1 = peak_idx1 / sig.shape[0] * samplingrate
    hr1 = f_hr1 * 60

    f_hr2 = peak_idx2 / sig.shape[0] * samplingrate
    hr2 = f_hr2 * 60
    if harmonics_removal:
        if np.abs(hr1 - 2 * hr2) < 10:
            hr = hr2
        else:
            hr = hr1
    else:
        hr = hr1

    x_psd = np.arange(len(sig)) / len(sig) * samplingrate * 60
    return hr, psd, x_psd


if __name__ == "__main__":
    pass