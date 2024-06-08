import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from runpy import run_path
from skimage import img_as_ubyte
import os

def save_img(filepath, img):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def load_img(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

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

def layer_images(file_, out_dir):
    # Load model architecture based on selected model
    parameters = {'inp_channels':3, 'out_channels':3, 'dim':48, 'num_blocks':[4,6,6,8], 'num_refinement_blocks':4, 'heads':[1,2,4,8], 'ffn_expansion_factor':2.66, 'bias':False, 'LayerNorm_type':'BiasFree', 'dual_pixel_task':False}
    load_arch = run_path("/hhome/ps2g07/document_analysis/github/Project_Synthesis2-/Code/Denoise/Restormer/basicsr/models/archs/restormer_6out_arch.py")
    model = load_arch['Restormer'](**parameters)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    weights = "/hhome/ps2g07/document_analysis/github/Project_Synthesis2-/Code/Denoise/Restormer/experiments/DocumentDenoisingLayers_ds_v4/models/zbest_psnr_l1.pth"
    checkpoint = torch.load(weights)
    model.load_state_dict(checkpoint['params'])
    model.eval()
    
    img_multiple_of = 8

    with torch.no_grad():
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

        input_ = load_img(file_)
        
        h, w = input_.shape[:2]
        if h > w:
            input_ = np.array(TF.resize(TF.to_pil_image(input_), (1000, int(1000*w/h))))
        else:
            input_ = np.array(TF.resize(TF.to_pil_image(input_), (int(1000*h/w), 1000)))
        input_ = torch.from_numpy(input_).float().div(255.).permute(2,0,1).unsqueeze(0).to(device)
        
        # Pad the input if not_multiple_of 8
        height, width = input_.shape[2], input_.shape[3]
        H, W = ((height+img_multiple_of)//img_multiple_of)*img_multiple_of, ((width+img_multiple_of)//img_multiple_of)*img_multiple_of
        padh = H-height if height%img_multiple_of!=0 else 0
        padw = W-width if width%img_multiple_of!=0 else 0
        input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

        restored = model(input_)
        restored_l0, restored_l1 = transform_output(restored)
        
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

        return restored_l0, restored_l1


if __name__ == "__main__":
    input_img = "/hhome/ps2g07/document_analysis/github/Project_Synthesis2-/Sample documents - JPG - NoName/1.jpg"
    out_dir = "/hhome/ps2g07/document_analysis/github/Project_Synthesis2-/output_images"
    
    restore_images(input_img, out_dir)