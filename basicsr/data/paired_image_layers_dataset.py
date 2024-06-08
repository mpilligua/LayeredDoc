from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_DP_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file)
from basicsr.data.transforms import augment, paired_random_crop, paired_random_crop_DP, random_augmentation
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, padding_DP, imfrombytesDP

import random
import numpy as np
import torch
import cv2
from os import path as osp
from basicsr.utils import scandir

def paired_paths_from_folder(folders, keys):
    """Generate paired paths from folders.

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder.

    Returns:
        list[str]: Returned path list.
    """
    
    input_folder, gt_l0_folder, gt_l1_folder = folders
    input_key, gt_l0_key, gt_l1_key = keys

    input_paths = list(scandir(input_folder))

    paths = []
    for idx in range(len(input_paths)):   
        input_path = input_paths[idx]
        
        gt_l0_path = osp.join(gt_l0_folder, input_path)
        gt_l1_path = osp.join(gt_l1_folder, input_path)
        input_path = osp.join(input_folder, input_path)
        
        paths.append({f'{input_key}_path': input_path,
                      f'{gt_l0_key}_path': gt_l0_path, 
                      f'{gt_l1_key}_path': gt_l1_path})
    return paths

def padding(img_lq, gt_size):
    h, w, _ = img_lq.shape

    h_pad = max(0, gt_size - h)
    w_pad = max(0, gt_size - w)
    
    if h_pad == 0 and w_pad == 0:
        return img_lq

    img_lq = cv2.copyMakeBorder(img_lq, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    
    if img_lq.ndim == 2:
        img_lq = np.expand_dims(img_lq, axis=2)
    return img_lq



def paired_random_crop(imgs, lq_patch_size, scale, gt_path):
    h_lq, w_lq, _ = imgs[0].shape
    gt_patch_size = int(lq_patch_size * scale)

    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    imgs_ret = []
    for img in imgs:
        if not isinstance(img, list):
            img = [img]

        # crop lq patch
        imgs_ret.append([v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img])

        if len(img) == 1:
            imgs_ret[-1] = imgs_ret[-1][0]

    return imgs_ret




class Dataset_PairedImage_Layers(data.Dataset):
    def __init__(self, opt):
        super(Dataset_PairedImage_Layers, self).__init__()
        self.opt = opt

        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        self.gt_l0_folder, self.gt_l1_folder, self.lq_folder = opt['dataroot_gt_l0'], opt['dataroot_gt_l1'], opt['dataroot_in']
        self.filename_tmpl = '{}'

        self.paths = paired_paths_from_folder([self.lq_folder, self.gt_l0_folder, self.gt_l1_folder], ['lq', 'gt_l0', 'gt_l1'])

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_l0_path = self.paths[index]['gt_l0_path']
        gt_l1_path = self.paths[index]['gt_l1_path']
        lq_path = self.paths[index]['lq_path']
        
        img_gt_l0 = self.open_img(gt_l0_path, 'gt_l0')
        img_gt_l1 = self.open_img(gt_l1_path, 'gt_l1')
        img_lq = self.open_img(lq_path, 'lq')

        # resize all images so that the largest side is 1000 pixels
        max_size = 1000
        h, w, _ = img_lq.shape
        if max(h, w) > max_size:
            if h > w:
                new_h, new_w = max_size, int(w / h * max_size)
            else:
                new_h, new_w = int(h / w * max_size), max_size
            img_gt_l0 = cv2.resize(img_gt_l0, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            img_gt_l1 = cv2.resize(img_gt_l1, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            img_lq = cv2.resize(img_lq, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        gt_size = self.opt['gt_size']
        # padding
        img_gt_l0 = padding(img_gt_l0, gt_size)
        img_gt_l1 = padding(img_gt_l1, gt_size)
        img_lq = padding(img_lq, gt_size)
        
        # augmentation for training
        if self.opt['phase'] == 'train':
            # random crop
            imgs = paired_random_crop([img_gt_l0, img_gt_l1, img_lq], gt_size, scale, gt_l0_path)
            img_gt_l0, img_gt_l1, img_lq = imgs

            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt_l0, img_gt_l1, img_lq = random_augmentation(img_gt_l0, img_gt_l1, img_lq)
            
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt_l0, img_gt_l1, img_lq = img2tensor([img_gt_l0, img_gt_l1, img_lq],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt_l0, self.mean, self.std, inplace=True)
            normalize(img_gt_l1, self.mean, self.std, inplace=True)
        
        # print("img_lq_shape", img_lq.shape, "gt_l0", img_gt_l0.shape, "gt_l1", img_gt_l1.shape, flush=True)
        
        return {
            'lq': img_lq,
            'gt_l0': img_gt_l0,
            'gt_l1': img_gt_l1,
            'lq_path': lq_path,
            'gt_l0_path': gt_l0_path,
            'gt_l1_path': gt_l1_path
        }

    def __len__(self):
        return len(self.paths)
    
    def open_img(self, path, key):
        img_bytes = self.file_client.get(path, key)
        try:
            img = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception(f"{key} path {path} not working")
        return img
    
# def collate_fn(batch):
#     """Collate function for paired image dataset.

#     Args:
#         batch (list): A list of data dicts.

#     Returns:
#         dict: A dict containing data.
#     """
#     data = {}
#     for key in batch[0].keys():
#         data[key] = torch.stack([d[key] for d in batch])
#     return data
        

if __name__ == "__main__":
    
    # name: TrainSet
    # type: Dataset_PairedImage_Layers
    # dataroot_gt_l0: /hhome/ps2g07/document_analysis/github/Project_Synthesis2-/Datasets/Synthetic_DS_v3/train_crops/l0
    # dataroot_gt_l1: /hhome/ps2g07/document_analysis/github/Project_Synthesis2-/Datasets/Synthetic_DS_v3/train_crops/l1
    # dataroot_in: /hhome/ps2g07/document_analysis/github/Project_Synthesis2-/Datasets/Synthetic_DS_v3/train_crops/all
    # geometric_augs: true

    # io_backend:
    #   type: disk

    # # data loader
    # use_shuffle: true
    # num_worker_per_gpu: 1
    # batch_size_per_gpu: 1

    # ### -------------Progressive training--------------------------
    # mini_batch_sizes: [8,5,4,2,1,1]             # Batch size per gpu   
    # iters: [92000,64000,48000,36000,36000,24000]
    # gt_size: 384   # Max patch size for progressive training
    # gt_sizes: [128,160,192,256,320,384]  # Patch sizes for progressive training.
    # ### ------------------------------------------------------------

    # ### ------- Training on single fixed-patch size 128x128---------
    # # mini_batch_sizes: [8]   
    # # iters: [300000]
    # # gt_size: 128   
    # # gt_sizes: [128]
    # ### ------------------------------------------------------------

    # dataset_enlarge_ratio: 1
    # prefetch_mode: ~
    
    opt = {
        'dataroot_gt_l0': '/hhome/ps2g07/document_analysis/github/Project_Synthesis2-/Datasets/Synthetic_DS_v3/train_crops/l0',
        'dataroot_gt_l1': '/hhome/ps2g07/document_analysis/github/Project_Synthesis2-/Datasets/Synthetic_DS_v3/train_crops/l1',
        'dataroot_in': '/hhome/ps2g07/document_analysis/github/Project_Synthesis2-/Datasets/Synthetic_DS_v3/train_crops/all',
        'geometric_augs': True,
        'io_backend': {'type': 'disk'},
        'phase': 'train',
        'use_shuffle': True,
        'num_worker_per_gpu': 2,
        'batch_size_per_gpu': 25,
        'gt_size': 384,
        'mini_batch_sizes': [8,5,4,2,1,1],
        'iters': [92000,64000,48000,36000,36000,24000],
        'gt_sizes': [128,160,192,256,320,384],
        'dataset_enlarge_ratio': 1,
        'prefetch_mode': None,
        'scale': 1

    }
    dataset = Dataset_PairedImage_Layers(opt)
    dataloader = data.DataLoader(dataset, batch_size=opt['batch_size_per_gpu'], shuffle=opt['use_shuffle'], num_workers=opt['num_worker_per_gpu'])
    for dataa in dataloader:
        print()
        print(dataa['lq'].shape)
        print(dataa['gt_l0'].shape)
        print(dataa['gt_l1'].shape)
    
    print("finished")