import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.models.image_restoration_model import ImageCleanModel
from basicsr.utils import get_root_logger, imwrite, tensor2img

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')

import os
import random
import numpy as np
import cv2
import torch.nn.functional as F
from functools import partial
    
class Mixing_Augment_3items:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, target2, input_):
        lam = self.dist.rsample((1,1)).item()
    
        r_index = torch.randperm(target.size(0)).to(self.device)
    
        target = lam * target + (1-lam) * target[r_index, :]
        target2 = lam * target2 + (1-lam) * target2[r_index, :]
        input_ = lam * input_ + (1-lam) * input_[r_index, :]
    
        return target, target2, input_

    def __call__(self, target, target2, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, target2, input_ = self.augments[augment](target, target2, input_)
        else:
            augment = random.randint(0, len(self.augments)-1)
            target, target2, input_ = self.augments[augment](target, target2, input_)
        return target, target2, input_

class DocLayerModel_DIBCO(ImageCleanModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(DocLayerModel_DIBCO, self).__init__(opt)
        self.mixing_flag = opt.get('mixing_flag', False)
        if self.mixing_flag:
            self.mixing_augmentation = Mixing_Augment_3items(opt['mixup_beta'], opt['use_identity'], self.device)
        self.ema_decay = 0

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(self.device)
        else:
            raise ValueError('pixel loss are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()


    def feed_train_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt_l0' in data:
            self.gt_l0 = data['gt_l0'].to(self.device)
        
        if 'gt_l1' in data:
            self.gt_l1 = data['gt_l1'].to(self.device)

        if self.mixing_flag:
            output = self.mixing_augmentation([self.gt_l0, self.gt_l1, self.lq])
            self.gt_l0, self.gt_l1, self.lq = output[0], output[1], output[2]
        # print("train", self.lq.shape, self.gt_l0.shape, self.gt_l1.shape, flush=True)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt_l0' in data:
            self.gt_l0 = data['gt_l0'].to(self.device)
        if 'gt_l1' in data:
            self.gt_l1 = data['gt_l1'].to(self.device)
        # print("val", self.lq.shape, self.gt_l0.shape, self.gt_l1.shape, flush=True)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        preds = self.net_g(self.lq)
        
        if not isinstance(preds, list):
            preds = [preds]

        pred_l0, pred_l1 = self.transform_output(preds)
        # print(preds[0].shape, pred_l0[0].shape, pred_l1[0].shape)

        self.output_l0 = pred_l0[-1]
        self.output_l1 = pred_l1[-1]

        loss_dict = OrderedDict()
        # pixel loss
        l_pix = 0.
        for p_l0 in pred_l0:
            l_pix += self.cri_pix(p_l0, self.gt_l0)
        loss_dict['l0_pix'] = l_pix

        #l_pix = 0.
        #for p_l1 in pred_l1:
        #    l_pix += self.cri_pix(p_l1, self.gt_l1)
        loss_dict['l1_pix'] = l_pix

        #l_pix = loss_dict['l0_pix'] #+ (loss_dict['l1_pix'] * 5)

        l_pix.backward()
        if self.opt['train']['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def nonpad_test(self, img=None):
        if img is None:
            img = self.lq      
        
        self.net_g.eval()
        with torch.no_grad():
            pred = self.net_g(img)

        pred_l0, pred_l1 = self.transform_output(pred)

        self.output_l0 = pred_l0 ## Aqui peque no fas []
        self.output_l1 = pred_l1
        
        self.net_g.train()

    def pad_test(self, window_size):        
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        self.nonpad_test(img)
        
        _, _, h, w = self.output_l0.size()
        self.output_l0 = self.output_l0[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]
        self.output_l1 = self.output_l1[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]
        
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric + '_' + n: 0
                for metric in self.opt['val']['metrics'].keys() for n in ['l0', 'l1']
            }
        pbar = tqdm(total=len(dataloader), unit='image')

        window_size = self.opt['val'].get('window_size', 0)

        if window_size:
            test = partial(self.pad_test, window_size)
        else:
            test = self.nonpad_test

        cnt = 0
        img_saved = 0
        for idx, val_data in tqdm(enumerate(dataloader)):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data)
            test()

            visuals = self.get_current_visuals()
            sr_l0_img = tensor2img([visuals['result_l0']], rgb2bgr=rgb2bgr)
            sr_l1_img = tensor2img([visuals['result_l1']], rgb2bgr=rgb2bgr)
            if 'gt_l0' in visuals:
                gt_l0_img = tensor2img([visuals['gt_l0']], rgb2bgr=rgb2bgr)
                del self.gt_l0
            
            if 'gt_l1' in visuals:
                gt_l1_img = tensor2img([visuals['gt_l1']], rgb2bgr=rgb2bgr)
                del self.gt_l1

            # tentative for out of GPU memory
            del self.lq
            del self.output_l0
            del self.output_l1
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    if not img_saved:
                        save_img_l0_path = osp.join(self.opt['path']['visualization'], img_name, f'{img_name}_{current_iter}_l0.png')
                        save_img_l1_path = osp.join(self.opt['path']['visualization'], img_name, f'{img_name}_{current_iter}_l1.png')
                        save_gt_l0_img_path = osp.join(self.opt['path']['visualization'], img_name, f'{img_name}_{current_iter}_gt_l0.png')
                        save_gt_l1_img_path = osp.join(self.opt['path']['visualization'], img_name, f'{img_name}_{current_iter}_gt_l1.png')
                    
                        imwrite(sr_l0_img[1], save_img_l0_path)
                        imwrite(sr_l1_img[1], save_img_l1_path)
                        imwrite(gt_l0_img[1], save_gt_l0_img_path)
                        imwrite(gt_l1_img[1], save_gt_l1_img_path)
                        img_saved = 1

                
                else:
                    save_img_l0_path = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}_l0.png')
                    save_img_l1_path = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}_l1.png')
                    save_gt_l0_img_path = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}_gt_l0.png')
                    save_gt_l1_img_path = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}_gt_l1.png')
                
                    imwrite(sr_l0_img, save_img_l0_path)
                    imwrite(sr_l1_img, save_img_l1_path)
                    imwrite(gt_l0_img, save_gt_l0_img_path)
                    imwrite(gt_l1_img, save_gt_l1_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                for name, opt_ in opt_metric.items():
                    metric_type = opt_.pop('type')
                    for i, (sr_l0, sr_l1, gt_l0, gt_l1) in enumerate(zip(sr_l0_img, sr_l1_img, gt_l0_img, gt_l1_img)):
                        if use_image:
                            self.metric_results[name + '_l0'] += getattr(metric_module, metric_type)(sr_l0, gt_l0, **opt_)
                            self.metric_results[name + '_l1'] += getattr(metric_module, metric_type)(sr_l1, gt_l1, **opt_)
                        else:
                            # print(visuals['result_l0'].shape, visuals['gt_l0'].shape, flush=True)
                            self.metric_results[name + '_l0'] += getattr(metric_module, metric_type)(visuals['result_l0'][i], visuals['gt_l0'][i], **opt_)
                            self.metric_results[name + '_l1'] += getattr(metric_module, metric_type)(visuals['result_l1'][i], visuals['gt_l1'][i], **opt_)
                        
                        cnt += 1
            pbar.update(1)

            if idx > (5000/20) and self.opt['is_train']:
                break

        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        return current_metric

    def transform_output(self, output):
        # given a list of vector of size HxWx6 return a list of 2 tensors of size HxWx3
        if isinstance(output, list):
            pred_l0 = []
            pred_l1 = []
            for out in output: 
                pred_l0.append(out[:, :3, :])
                pred_l1.append(out[:, 3:, :])
            return pred_l0, pred_l1
        else: 
            return output[:, :3, :], output[:, 3:, :]
    
    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result_l0'] = self.output_l0.detach().cpu()
        out_dict['result_l1'] = self.output_l1.detach().cpu()
        
        if hasattr(self, 'gt_l0'):
            out_dict['gt_l0'] = self.gt_l0.detach().cpu()
            
        if hasattr(self, 'gt_l1'):
            out_dict['gt_l1'] = self.gt_l1.detach().cpu()
        return out_dict