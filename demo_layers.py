## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

##--------------------------------------------------------------
##------- Demo file to test Restormer on your own images---------
## Example usage on directory containing several images:   python demo.py --task Single_Image_Defocus_Deblurring --input_dir './demo/degraded/' --result_dir './demo/restored/'
## Example usage on a image directly: python demo.py --task Single_Image_Defocus_Deblurring --input_dir './demo/degraded/portrait.jpg' --result_dir './demo/restored/'
## Example usage with tile option on a large image: python demo.py --task Single_Image_Defocus_Deblurring --input_dir './demo/degraded/portrait.jpg' --result_dir './demo/restored/' --tile 720 --tile_overlap 32
##--------------------------------------------------------------

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import os
from runpy import run_path
from skimage import img_as_ubyte
from natsort import natsorted
from glob import glob
import cv2
from tqdm import tqdm
import argparse
from pdb import set_trace as stx
import numpy as np

parser = argparse.ArgumentParser(description='Test Restormer on your own images')

#parser.add_argument('--input_dir', default='/hhome/ps2g07/document_analysis/github/Project_Synthesis2-/Sample documents - JPG - NoName', type=str, help='Directory of input images or path of single image')
parser.add_argument('--input_dir', default='/ghome/mpilligua/DocumentConversion/Data/NewTest/ALL', type=str, help='Directory of input images or path of single image')

parser.add_argument('--result_dir', default='/ghome/mpilligua/DocumentConversion/Results_paper/Inferencia/Restormer-6Ch_lossChanged/Millor_l1/NewTest', type=str, help='Directory for restored results')
parser.add_argument('--task', required=False, default='Real_Denoising', type=str, help='Task to run', choices=['Motion_Deblurring',
                                                                                    'Single_Image_Defocus_Deblurring',
                                                                                    'Deraining',
                                                                                    'Real_Denoising',
                                                                                    'Gaussian_Gray_Denoising',
                                                                                    'Gaussian_Color_Denoising'])

parser.add_argument('--tile', type=int, default=None, help='Tile size (e.g 720). None means testing on the original resolution image')
parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
parser.add_argument('--ckpt_dir', required=False, type=str, default='/ghome/mpilligua/DocumentConversion/Denoise/Restormer/experiments/Restormer_6Channels_lossChanged/models/zbest_psnr_l1.pth', help='Directory of the model checkpoint')

args = parser.parse_args()

def load_img(filepath):
    print(filepath)
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def load_gray_img(filepath):
    return np.expand_dims(cv2.imread(filepath, cv2.IMREAD_GRAYSCALE), axis=2)

def save_gray_img(filepath, img):
    cv2.imwrite(filepath, img)

def get_weights_and_parameters(parameters, ckpt_dir):
    weights = ckpt_dir
    parameters['LayerNorm_type'] =  'BiasFree'
    
    return weights, parameters

def transform_output(output):
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

task    = args.task
inp_dir = args.input_dir
out_dir = os.path.join(args.result_dir, task)

os.makedirs(out_dir, exist_ok=True)

extensions = ['jpg', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']

if any([inp_dir.endswith(ext) for ext in extensions]):
    print("a")
    files = [inp_dir]
else:
    files = []
    for ext in extensions:
        print("Ge")
        print(inp_dir)
        files.extend(glob(os.path.join(inp_dir, '*.'+ext)))
        print(files)
    files = natsorted(files)

print("Using 250 images")
files = files[:250]
if len(files) == 0:
    raise Exception(f'No files found at {inp_dir}')

# Get model weights and parameters
parameters = {'inp_channels':3, 'out_channels':6, 'dim':48, 'num_blocks':[4,6,6,8], 'num_refinement_blocks':4, 'heads':[1,2,4,8], 'ffn_expansion_factor':2.66, 'bias':False, 'LayerNorm_type':'WithBias', 'dual_pixel_task':False}

#ckpt_dir = 'DocumentConversion/Denoise/Restormer/experiments/DocumentDenoisingLayers_ds_v5/models/zbest_psnr_l0.pth'
#weights, parameters = get_weights_and_parameters(parameters, ckpt_dir)
weights, parameters = get_weights_and_parameters(parameters, args.ckpt_dir)

load_arch = run_path(os.path.join('basicsr', 'models', 'archs', 'restormer_arch.py'))
model = load_arch['Restormer'](**parameters)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

checkpoint = torch.load(weights)
model.load_state_dict(checkpoint['params'])
model.eval()

img_multiple_of = 8

print(f"\n ==> Running {task} with weights {weights}\n ")


with torch.no_grad():
    for file_ in tqdm(files):
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

        if task == 'Gaussian_Gray_Denoising':
            img = load_gray_img(file_)
        else:
            img = load_img(file_)
            
            # resize the image so that the bigger dimension is 1000
            h, w = img.shape[:2]
            if h > w:
                img = np.array(TF.resize(TF.to_pil_image(img), (1000, int(1000*w/h))))
            else:
                img = np.array(TF.resize(TF.to_pil_image(img), (int(1000*h/w), 1000)))

        input_ = torch.from_numpy(img).float().div(255.).permute(2,0,1).unsqueeze(0).to(device)

        # Pad the input if not_multiple_of 8
        height,width = input_.shape[2], input_.shape[3]
        H,W = ((height+img_multiple_of)//img_multiple_of)*img_multiple_of, ((width+img_multiple_of)//img_multiple_of)*img_multiple_of
        padh = H-height if height%img_multiple_of!=0 else 0
        padw = W-width if width%img_multiple_of!=0 else 0
        input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

        if args.tile is None:
            ## Testing on the original resolution image
            restored = model(input_)
            restored_l0, restored_l1 = transform_output(restored)
            #print(restored_l0.shape, restored_l1.shape)
        # else:
        #     # test the image tile by tile
        #     b, c, h, w = input_.shape
        #     tile = min(args.tile, h, w)
        #     assert tile % 8 == 0, "tile size should be multiple of 8"
        #     tile_overlap = args.tile_overlap

        #     stride = tile - tile_overlap
        #     h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        #     w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        #     E = torch.zeros(b, c, h, w).type_as(input_)
        #     W = torch.zeros_like(E)

        #     for h_idx in h_idx_list:
        #         for w_idx in w_idx_list:
        #             in_patch = input_[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
        #             out_patch = model(in_patch)
        #             out_patch_mask = torch.ones_like(out_patch)

        #             E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch)
        #             W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_mask)
        #     restored = E.div_(W)

        restored_l0 = torch.clamp(restored_l0, 0, 1)
        restored_l1 = torch.clamp(restored_l1, 0, 1)

        # Unpad the output
        restored_l0 = restored_l0[:,:,:height,:width]
        restored_l1 = restored_l1[:,:,:height,:width]

        restored_l0 = restored_l0.permute(0, 2, 3, 1).cpu().detach().numpy()
        restored_l1 = restored_l1.permute(0, 2, 3, 1).cpu().detach().numpy()
        
        restored_l0 = img_as_ubyte(restored_l0[0])
        restored_l1 = img_as_ubyte(restored_l1[0])

        f = os.path.splitext(os.path.split(file_)[-1])[0]
        save_img((os.path.join(out_dir, f+'_l0.png')), restored_l0)
        save_img((os.path.join(out_dir, f+'_l1.png')), restored_l1)

    print(f"\nRestored images are saved at {out_dir}")
