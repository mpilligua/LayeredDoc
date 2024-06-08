## Learning Enriched Features for Fast Image Restoration and Enhancement
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)
## https://www.waqaszamir.com/publication/zamir-2022-mirnetv2/

import cv2
import torch
import numpy as np
from glob import glob
from natsort import natsorted
import os
from tqdm import tqdm
from pdb import set_trace as stx

split = ""

root = '/ghome/mpilligua/DocumentConversion/data/DIBCOSETS/training_2009-2019' + split

src_gt_l0 = root + '/gt_l0'
src_gt_l1 = root + '/gt_l1'
src_gt_all = root + '/gt_all'

tar = root + '_crops'

import json
import pandas as pd

# annot_path = '/hhome/ps2g07/document_analysis/github/Project_Synthesis2-/Datasets/Synthetic_DS/train.json'
# with open(annot_path) as f:
#     data = json.load(f)

# annotations = pd.DataFrame(data["annotations"])
# images = pd.DataFrame(data["images"])
# # merge both dataframes on image_id
# annotations = pd.merge(annotations, images, how='left', left_on='image_id', right_on='id')
# docs_with_tables = annotations.groupby(['image_id']).filter(lambda x: (x['category_id'] == 4).any())['file_name'].to_list()

l0_tar = os.path.join(tar, 'l0')
l1_tar = os.path.join(tar, 'l1')
all_tar = os.path.join(tar, 'all')

os.makedirs(l0_tar, exist_ok=True)
os.makedirs(l1_tar, exist_ok=True)
os.makedirs(all_tar, exist_ok=True)

files = natsorted(glob(os.path.join(src_gt_l0, '*.png')))
print(files)

l0_files, l1_files, all_files = [], [], []
for file_ in files:
    # if file_.split('/')[-1] not in docs_with_tables:
    #     continue
    # filename = os.path.split(file_)[-1]
    l0_files.append(file_)
    l1_files.append(file_.replace(src_gt_l0, src_gt_l1))
    all_files.append(file_.replace(src_gt_l0, src_gt_all))
    

# print
files = [(i, j, k) for i, j, k in zip(l0_files, l1_files, all_files)]
# print(files)
patch_size = 512
overlap = 128
p_max = 0

def save_files(file_):
    l0_file, l1_file, all_file = file_
    filename = os.path.splitext(os.path.split(l0_file)[-1])[0]
    l0_img = cv2.imread(l0_file)
    l1_img = cv2.imread(l1_file)
    all_img = cv2.imread(all_file)
    num_patch = 0
    w, h = l0_img.shape[:2]
    if w > p_max and h > p_max:
        w1 = list(np.arange(0, w-patch_size, patch_size-overlap, dtype=int))
        h1 = list(np.arange(0, h-patch_size, patch_size-overlap, dtype=int))
        w1.append(w-patch_size)
        h1.append(h-patch_size)
        for i in w1:
            for j in h1:
                num_patch += 1
                
                l0_patch = l0_img[i:i+patch_size, j:j+patch_size,:]
                l1_patch = l1_img[i:i+patch_size, j:j+patch_size,:]
                all_patch = all_img[i:i+patch_size, j:j+patch_size,:]
                
                l0_savename = os.path.join(l0_tar, filename + '-' + str(num_patch) + '.png')
                l1_savename = os.path.join(l1_tar, filename + '-' + str(num_patch) + '.png')
                all_savename = os.path.join(all_tar, filename + '-' + str(num_patch) + '.png')
                
                cv2.imwrite(l0_savename, l0_patch)
                cv2.imwrite(l1_savename, l1_patch)
                cv2.imwrite(all_savename, all_patch)

    else:
        l0_savename = os.path.join(l0_tar, filename + '.png')
        l1_savename = os.path.join(l1_tar, filename + '.png')
        all_savename = os.path.join(all_tar, filename + '.png')
        
        cv2.imwrite(l0_savename, l0_img)
        cv2.imwrite(l1_savename, l1_img)
        cv2.imwrite(all_savename, all_img)

from joblib import Parallel, delayed
import multiprocessing
num_cores = 10
Parallel(n_jobs=num_cores)(delayed(save_files)(file_) for file_ in tqdm(files))
