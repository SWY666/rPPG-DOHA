# from .PURE import Dataset_PURE_Offline
# from .UBFC import Dataset_UBFC_Offline
from .dataset_VIPL import Dataset_VIPL_HR_Offline
import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# from torchvision import transforms
# import cv2 as cv
from PIL import Image
# import albumentations as A
# from utils.ippg_attn_make import self_similarity_calc

global_root_dir = r"./data"

# def return_dataset_PURE(person_name, version_type, is_train=True, root_dir=global_root_dir, length=300, frame_drop=12, leave_1=False, length_train=70):
#     Dataset_place = os.path.join(root_dir, "pure-frame-v2")
#     frame_palce = Dataset_place + r"/frame_list"
#     mask_path = Dataset_place + r"/mask_list"
#     wave_bvp_path = Dataset_place + r"/wave_bvp"
#     wave_path = Dataset_place + r"/wave_gt"
#     dataset = Dataset_PURE_Offline(frame_palce, person_name, version_type, mask_path
#                                    , wave_path, wave_bvp_path, length=length_train if is_train else length, is_train=is_train, frame_drop=frame_drop, leave_1=leave_1)
#     return dataset


def return_dataset_VIPL(person_name, version_type, is_train=True, root_dir=global_root_dir, length=300, frame_drop=12, leave_1=False, length_train=70, data_dir=r"", dataset_place=""):
    Dataset_place = os.path.join(root_dir, "vipl-frame" if data_dir == "" else data_dir) if dataset_place == "" else dataset_place
    frame_palce = Dataset_place + r"/frame_list"
    mask_path = Dataset_place + r"/mask_list"
    wave_path = Dataset_place + r"/wave_gt"
    dataset = Dataset_VIPL_HR_Offline(frame_palce, person_name, version_type, mask_path
                                      , wave_path, length=length_train if is_train else length, is_train=is_train, frame_drop=frame_drop, leave_1=leave_1)
    return dataset


# def return_dataset_UBFC(person_name, version_type, is_train=True, root_dir=global_root_dir, length=300, frame_drop=12, leave_1=False, length_train=70):
#     Dataset_place = os.path.join(root_dir, "ubfc-frame-v2")
#     frame_palce = Dataset_place + r"/frame_list"
#     mask_path = Dataset_place + r"/mask_list"
#     wave_bvp_path = Dataset_place + r"/wave_bvp"
#     wave_path = Dataset_place + r"/wave_gt"
#     dataset = Dataset_UBFC_Offline(frame_palce, person_name, mask_path
#                                    , wave_path, wave_bvp_path, length=length_train if is_train else length, is_train=is_train, frame_drop=frame_drop, leave_1=leave_1)
#     return dataset