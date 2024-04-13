import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# from torchvision import transforms
import cv2 as cv
from PIL import Image
import albumentations as A
from utils.ippg_attn_make import self_similarity_calc

class Dataset_VIPL_HR_Offline(Dataset):
    def __init__(self, frame_path, person_name, version_type, mask_path, wave_path, frame_drop=12, length=70, is_train=True):
        super().__init__()
        self.length = length
        self.frame_drop = frame_drop
        self.person_name = person_name
        self.version_type = version_type
        self.frame_list_root_path = frame_path
        # self.map_gt_root_path = map_path
        self.mask_list_root_path = mask_path
        self.wave_gt_root_path = wave_path
        self.frame_lists_total = os.listdir(self.frame_list_root_path)
        # self.map_gts_total = os.listdir(self.map_gt_root_path)
        self.mask_lists_total = os.listdir(self.mask_list_root_path)
        self.wave_gts_total = os.listdir(self.wave_gt_root_path)
        self.frame_lists = []
        self.mask_lists = []
        self.wave_gts = []
        self.is_train = is_train
        self.prepare_data()

    def __getitem__(self, index):
        idx = index % len(self.frame_lists)
        frame_list_path = os.path.join(self.frame_list_root_path, self.frame_lists[idx])
        mask_list_path = os.path.join(self.mask_list_root_path, self.mask_lists[idx])
        wave_gt_path = os.path.join(self.wave_gt_root_path, self.wave_gts[idx])

        frame_list_origin = np.load(frame_list_path)
        mask_list_origin = np.load(mask_list_path)
        wave_return_new = np.load(wave_gt_path)

        frame_list_tf = np.zeros([frame_list_origin.shape[0], 131, 131, 3], dtype=np.uint8)
        mask_list_tf = np.zeros([mask_list_origin.shape[0], 131, 131], dtype=np.uint8)

        ## Data Augmentation, only for training.
        if self.is_train:
            # 1.random crop.
            dh, dw = self.random_shake_frame(margin=20)
            frame_list_origin = frame_list_origin[:, dh:131 + dh, dw:131 + dw, :]
            mask_list_origin = mask_list_origin[:, dh:131 + dh, dw:131 + dw]

            # 2.random flip.
            is_hFlip = np.random.random() > 0.5
            is_vFlip = np.random.random() > 0.5
            transform = A.Compose([
                A.VerticalFlip(p=is_hFlip),
                A.HorizontalFlip(p=is_vFlip)
            ])
            for i in range(frame_list_origin.shape[0]):
                if i == frame_list_origin.shape[0] - 1:
                    frame_list_tf[i] = transform(image=frame_list_origin[i])['image']
                else:
                    transformed = transform(image=frame_list_origin[i], mask=mask_list_origin[i])
                    frame_list_tf[i] = transformed['image']
                    mask_list_tf[i] = transformed['mask']

            # 3.random temporal up-sampling/down-sampling based on HR.
            hr_gt = float(self.frame_lists[idx].split('_')[6])
            is_sample = np.random.random() > 0.5
            if is_sample:
                if hr_gt > 90:  # halve hr -> up-sampling.
                    # frame & mask keep left-aligned:
                    # frame: ⚪ ⚪ ⚪ ⚪ ⚪ -> ⚪ × ⚪ × ⚪ × ⚪ × ⚪
                    # mask:  ⚪ ⚪ ⚪ ⚪ -> ⚪ × ⚪ × ⚪ × ⚪ ×
                    # wave:  ⚪ ⚪ ⚪ ⚪ ⚪ -> ⚪ × ⚪ × ⚪ × ⚪ × ⚪
                    frame_remastered = np.zeros([frame_list_tf.shape[0]*2 - 1, frame_list_tf.shape[1],
                                                 frame_list_tf.shape[2], frame_list_tf.shape[3]], dtype=np.uint8)
                    mask_remastered = np.zeros([mask_list_tf.shape[0]*2, mask_list_tf.shape[1],
                                                mask_list_tf.shape[2]], dtype=np.uint8)
                    wave_remastered = np.zeros(len(wave_return_new)*2 - 1)
                    for i in range(frame_remastered.shape[0]):
                        if i % 2 == 0:
                            frame_remastered[i, :, :, :] = frame_list_tf[i//2, :, :, :]
                        else:
                            frame_remastered[i, :, :, :] = frame_list_tf[i//2, :, :, :]//2 + frame_list_tf[i//2+1, :, :, :]//2

                    for i in range(mask_remastered.shape[0]):
                        if i != mask_remastered.shape[0] - 1:
                            if i % 2 == 0:
                                mask_remastered[i, :, :] = mask_list_tf[i//2, :, :]
                            else:
                                # do not use '//2' for interpolation.
                                mask_remastered[i, :, :] = mask_list_tf[i//2, :, :]/2 + mask_list_tf[i//2+1, :, :]/2
                        else:
                            # the last one does not matter, we will not sample it...
                            mask_remastered[i, :, :] = mask_remastered[i-1, :, :]//2 + mask_remastered[i-2, :, :]//2

                    for i in range(wave_remastered.shape[0]):
                        if i % 2 == 0:
                            wave_remastered[i] = wave_return_new[i//2]
                        else:
                            wave_remastered[i] = 0.5*wave_return_new[i//2] + 0.5*wave_return_new[i//2+1]

                    frame_list_tf = frame_remastered
                    mask_list_tf = mask_remastered
                    wave_return_new = wave_remastered

                # double hr -> down-sampling.
                elif hr_gt < 70 and \
                        int(self.frame_lists[idx].split('_')[5]) - int(self.frame_lists[idx].split('_')[4]) > self.length*2:
                    # frame & mask keep left-aligned:
                    # frame: ⚪ ⚪ ⚪ ⚪ ⚪ -> ⚪ ⚪ ⚪
                    # mask:  ⚪ ⚪ ⚪ ⚪ -> ⚪ ⚪
                    # wave:  ⚪ ⚪ ⚪ ⚪ ⚪ -> ⚪ ⚪ ⚪
                    frame_remastered = np.zeros([(frame_list_tf.shape[0]+1)//2, frame_list_tf.shape[1],
                                                 frame_list_tf.shape[2], frame_list_tf.shape[3]], dtype=np.uint8)
                    mask_remastered = np.zeros([mask_list_tf.shape[0]//2, mask_list_tf.shape[1],
                                                mask_list_tf.shape[2]], dtype=np.uint8)
                    wave_remastered = np.zeros((len(wave_return_new)+1)//2)

                    for i in range(frame_remastered.shape[0]):
                        frame_remastered[i, :, :, :] = frame_list_tf[i*2, :, :, :]

                    for i in range(mask_remastered.shape[0]):
                        mask_remastered[i, :, :] = mask_list_tf[i*2, :, :]

                    for i in range(wave_remastered.shape[0]):
                        wave_remastered[i] = wave_return_new[i*2]

                    frame_list_tf = frame_remastered
                    mask_list_tf = mask_remastered
                    wave_return_new = wave_remastered

                else:  # too short for this.
                    pass

        else:
            # center crop for validation.
            frame_list_tf = frame_list_origin[:, 10:141, 10:141, :]
            mask_list_tf = mask_list_origin[:, 10:141, 10:141]

        mask_list = np.zeros([mask_list_tf.shape[0], 64, 64], dtype=np.uint8)
        residual_list = np.zeros([frame_list_tf.shape[0] - 1, frame_list_tf.shape[1], frame_list_tf.shape[2], frame_list_tf.shape[3]], dtype=np.int16)
        for i in range(residual_list.shape[0]):
            residual_list[i, :, :, :] = frame_list_tf[i+1, :, :, :].astype(np.int16) - frame_list_tf[i, :, :, :].astype(np.int16)  # (frame_length-1, 131, 131, 3)
            # mask: 131 -> 64
            mask_list[i] = cv.resize(mask_list_tf[i], (64, 64))

        ## ndarray -> Tensor
        frame_list = self.My_FloatTensor(frame_list_tf).permute((3, 0, 1, 2))
        mask_list = self.My_FloatTensor(mask_list)
        residual_list = self.My_FloatTensor(residual_list).permute((3, 0, 1, 2))

        ## training sample generate...
        if self.length == frame_list.shape[1]:
            start = 0
        else:
            start = np.random.randint(0, frame_list.shape[1] - self.length)
        end = start + self.length

        residual_list_return = residual_list[:, start:end-1, :, :]
        frame_list_return = frame_list[:, start:end, :, :]
        mask_list_return = mask_list[start:end-1, :, :]
        wave_gt_return = wave_return_new[start:end]
        _wave_drop = wave_gt_return[self.frame_drop: -self.frame_drop]
        map_gt_return = self_similarity_calc(_wave_drop)
        name = '_'.join([self.frame_lists[idx].split('_')[0], self.frame_lists[idx].split('_')[1], self.frame_lists[idx].split('_')[2],
                         self.frame_lists[idx].split('_')[4], str(int(self.frame_lists[idx].split('_')[4]) + self.length)])
        return residual_list_return, frame_list_return, map_gt_return, mask_list_return, wave_gt_return, self.frame_lists[idx], start, end, name

    def __len__(self):
        return len(self.frame_lists)

    @staticmethod
    def random_shake_frame(margin):
        dh = 0
        dw = 0
        choice = np.random.random()
        if 0 <= choice < 0.3:
            change_size = np.random.randint(0, margin)
            dh = change_size
        elif 0.3 <= choice < 0.6:
            change_size = np.random.randint(0, margin)
            dw = change_size
        else:
            change_size = np.random.randint(0, margin)
            dh = change_size
            dw = change_size
        return dh, dw

    def prepare_data(self):
        for path in self.frame_lists_total:
            if path.split('_')[1] in self.person_name and path.split('_')[2] in self.version_type\
                    and int(path.split('_')[5]) - int(path.split('_')[4]) >= self.length:
                self.frame_lists.append(path)
        for path in self.mask_lists_total:
            if path.split('_')[1] in self.person_name and path.split('_')[2] in self.version_type\
                    and int(path.split('_')[5]) - int(path.split('_')[4]) >= self.length:
                self.mask_lists.append(path)
        for path in self.wave_gts_total:
            if path.split('_')[1] in self.person_name and path.split('_')[2] in self.version_type\
                    and int(path.split('_')[5]) - int(path.split('_')[4]) >= self.length:
                self.wave_gts.append(path)
        self.frame_lists.sort(key=lambda x: int(x.split('_')[0]))
        # self.map_gts.sort(key=lambda x: int(x.split('_')[0]))
        self.mask_lists.sort(key=lambda x: int(x.split('_')[0]))
        self.wave_gts.sort(key=lambda x: int(x.split('_')[0]))

    @staticmethod
    def My_FloatTensor(array):
        # Turn list with frames into Tensor in python
        length_of_list = len(array)
        puzzle_list = [torch.FloatTensor(array[item]).unsqueeze(0) for item in range(length_of_list)]
        result = torch.cat(puzzle_list, dim=0)
        return result

    @staticmethod
    def My_IntTensor(array):
        # Turn list with frames into Tensor in python
        length_of_list = len(array)
        puzzle_list = [torch.IntTensor(array[item]).unsqueeze(0) for item in range(length_of_list)]
        result = torch.cat(puzzle_list, dim=0)
        return result



