# DOHA-rPPG (DOmain-HArmonious framework)
This is the code corresponding to the paper ["Resolve Domain Conflicts for Generalizable Remote Physiological Measurement."](https://dl.acm.org/doi/10.1145/3581783.3612265) accepted in ACM MM 2023. :fire: :fire: :fire:

## Recent updates

**2024.Apr.13**  Provide a simple demonstration on how to train a rPPG model based on our methodology. You can refer it in ***"Quick Start of our methodology"*** session!


## Quick Start of our methodology

### **Quick start**

You can run an example of our method by executing the following command. This code utilizes **perticipant 1 (p1)** in VIPL-HR dataset [1] for training the model in (**models/model.py (Ultimate_model)**!) You can see the default hyper-parameter setting in the ***get_args_parser*** function in **quick_start.py**!

```
> python quick_start.py --dataset_place  ./mini_vipl_data/vipl-frame-mini --GPU-id 0
```
### Introduction to our specialized dataloader (may vary between different rPPG network trainers)

In brief, the ouput format of our data loader is shown as follows:

```
for it, (inputs, real_image_train, train_attn_label, skin_mask, wave_label, path, start_f, end_f, _, index) in enumerate(data_loader):
```

where the **real_image_train** (shape: [batch size, 3, frame_length, width, height]) is the face region of interest extracted from the original facial video; **inputs** is the residual version of **real_image_train** (you can refer to Deepphys [2] to know about the residual version, in brief, **inputs** = **real_image_train[:,:,1:,:,:]-real_image_train[:,:,:-1,:,:]**); **wave_label** is the corresponding ground truth label of **real_image_train**; **wave_label** has **an uncertain temporal delay** between **real_image_train** due to pulse transit time and system error; **train_attn_label** is the ST-MAP version of **wave_label**, you can utilize function ***self_similarity_calc*** to turn **wave_label** into **train_attn_label**, which is shown as follows:

```
train_attn_label = self_similarity_calc(wave_label) # this function can be found in utils/ippg_attn_make.py!
```
### Advice to trainers with different datasets and model backbones

We utilize the TS-CAN as the backbone model (which requires both original and residual inputs), hence we need both **real_image_train** and **inputs**. For those trainers who utilize another pipeline, it is recommended that you can switch our dataloader that suits you best (note: Ground truth label (**train_attn_label**, which is used for our model training) should be kept for training the model via our method)!

[1] VIPL-HR: A Multi-modal Database for Pulse Estimation from Less-constrained Face Video

[2] "DeepPhys: Video-Based Physiological Measurement Using Convolutional Attention Networks" https://arxiv.org/abs/1805.07888.


## What is rPPG? :crossed_fingers: :crossed_fingers: :crossed_fingers:
Generally speaking, remote photoplethysmography (rPPG) is a special  kind of physiological measurement which measures the physiological information (such as heart rate) of the subject through his/her facial video. The overall workflow of a typical rPPG is shown as follows:

![What is rPPG?](img_bank/What_is_rPPG.png "rPPG")

The key element of rPPG is to extract the so-called **rPPG signal** from the facial video. The rPPG signal is regarded to presented the similar physiological features with **Blood Volume Pulse (BVP) signal**, from which we can extract various physiological information, such as heart rate and heart rate variation (**maybe?**).


## Current domain conflict Issues of DNN rPPG model training. :love_you_gesture: :love_you_gesture: :love_you_gesture:
Related databases for rPPG model training have the following flaws. (1) **Label Conflict**: The ground truth rPPG signals (e.g., BVP signal) in the database are not uniformed in time, i.e., there exist a unknown temporal delay between label and truth rPPG signal. (2) **Attribute Conflict**: The individual variability of data (e.g., skin color, ambient light) renders the rPPG model inefficient to train. We illustrate above two issues as follows:

<div align=center><img src="img_bank/ACM_MM_2023_Heart_Rate_seal.png" width="60%" ></div>

## Harmonious Phase Strategy. :call_me_hand: :call_me_hand: :call_me_hand:

We focus on the self-similarity property of rPPG signal (which is periodical approximately), and thus generate a novel label representation, namely self-cosine-similarity (SCS) map, shown as follows:

<div align=center><img src="img_bank/SSP_map.png" width="50%" ></div>

By replacing the traditional ground truth label to our proposed SCS map for rPPG model training, we can elegantly get rid of label conflicts (as our new label representation doesn’t involve the real-time temporal delay). A typical network training pipeline is shown as follows:

<div align=center><img src="img_bank/HPS.png" width="80%" ></div>

**Note**: When selecting the ground truth signal (e.g., BVP signal), we choose the **identical timestamps** that "match" the input video. For example, if we select the video slice (from 10s to 20s) in the video, we pick the ground truth signal (from 10s to 20s) too. **Noteworthy**, even though we match the timestamps of the input video and the ground truth signal, we still can't ensure that these two are aligned due to the uncertain temporal delay. **Nevertheless**, our method can neutralize such a temporal delay, making the training more accessible and efficient.


## Still under construction! 😍 😍 😍
Thanks for your reading! If this project is helpful to you, we hope you can cite our work as follows! 🥳🥳🥳

```
@inproceedings{10.1145/3581783.3612265,author = {Sun, Weiyu and Zhang, Xinyu and Lu, Hao and Chen, Ying and Ge, Yun and Huang, Xiaolin and Yuan, Jie and Chen, Yingcong},
title = {Resolve Domain Conflicts for Generalizable Remote Physiological Measurement},
year = {2023},
isbn = {9798400701085},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3581783.3612265},
doi = {10.1145/3581783.3612265},
booktitle = {Proceedings of the 31st ACM International Conference on Multimedia},
pages = {8214–8224},
numpages = {11},
keywords = {physiological signal estimation, multimedia application, rppg},
location = {Ottawa ON, Canada},
series = {MM '23}
}
```