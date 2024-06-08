import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
import pyiqa
# from utils.our_utils import *
# from base_parser import BaseParser
import pandas as pd
from skimage import color
from skimage.color import deltaE_cie76
import torch
from tqdm import tqdm

# from UNIQUE.BaseCNN import BaseCNN
# from UNIQUE.Main import parse_config
# from UNIQUE.Transformers import AdaptiveResize


import os
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
import torch

# from NIMA.model.model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

psnr = pyiqa.create_metric('psnr', test_y_channel=False).to(device)
ilum_psnr = pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr').to(device)
ssim = pyiqa.create_metric('ssim').to(device)
lpips = pyiqa.create_metric('lpips').to(device)
niqe = pyiqa.create_metric('niqe').to(device)
brisque = pyiqa.create_metric('brisque').to(device)
hyperiqa = pyiqa.create_metric('hyperiqa').to(device)

I2sensor = {'Red': {254: 1002.4110000000001, 241: 902.8583, 227: 801.4689, 213: 706.1163, 197: 604.5339, 180: 505.242, 161: 404.8023, 139: 302.3945, 113: 200.58630000000002, 80: 101.352, 0: 49}, 'Green': {254: 1926.4376, 241: 1734.1636999999998, 227: 1538.4017000000001, 213: 1354.3604999999998, 197: 1158.3797, 180: 966.924, 161: 773.3957, 139: 576.2801, 113: 380.6405, 80: 190.54399999999998, 0: 49}, 'Blue': {254: 193.59879999999998, 241: 173.97789999999998, 227: 154.0195, 213: 135.2763, 197: 115.34349999999999, 180: 95.904, 161: 76.2979, 139: 56.3923, 113: 36.7363, 80: 17.823999999999998, 0: 49}, 'White': {254: 1861.312, 241: 1675.5043, 227: 1486.3278999999998, 213: 1308.4803, 197: 1119.0979, 180: 934.0919999999999, 161: 747.0883, 139: 556.6255, 113: 367.60029999999995, 80: 183.95199999999997, 0: 49}}
IList = [0, 80, 113, 139, 161, 180, 197, 213, 227, 241, 254]

def compute_metrics(opt):
    print("Computing metrics for", "in", opt["pred_dir"])

    out_dir = opt["save_dir"]
    M = pd.DataFrame(index = [[], []])
    for file in tqdm(os.listdir(opt["pred_dir"])):
        print(file)
        if "metrics" not in file and "l1" not in file and "denuncia" not in file:
            path_pred = opt["pred_dir"] + "/" + file
            name_true = file.replace('_l0', '')
            # try:
            if "matricula" not in file:
                path_true = opt["high_dir"] + "/" + name_true
            else:
                path_true = opt["high_dir"] + "/" + name_true

            if os.path.isdir(path_pred) or os.path.isdir(path_true):
                continue

            pil_pred = Image.open(path_pred)
            pil_true = Image.open(path_true)
            # except:
            #     if "matricula" not in file:
            #         path_true = opt["high_dir"] + "/" + name_true + ".png"
            #     else:
            #         path_true = opt["high_dir"] + "/" + name_true + ".png"
            #     pil_pred = Image.open(path_pred)
            #     pil_true = Image.open(path_true)

            if len(pil_pred.size) == 2:
                pil_pred = pil_pred.convert('RGB')
                
            if len(pil_true.size) == 2:
                pil_true = pil_true.convert('RGB')

            pil_pred = pil_pred.resize((pil_true.size[0], pil_true.size[1]))

            pil_pred = Image.fromarray(np.array(pil_pred))
            ## Normalize the images

            y_pred = np.array(pil_pred) 
            y_true = np.array(pil_true)

            global psnr, ssim, lpips, niqe, brisque, hyperiqa

            M.loc[(file), 'MSE ↓'] = np.mean((y_pred - y_true) ** 2)
            M.loc[(file), 'PSNR(color) ↑'] = psnr(pil_pred, pil_true).item()
            M.loc[(file), 'PSNR(ilum) ↑'] = ilum_psnr(pil_pred, pil_true).item()
            M.loc[(file), 'SSIM ↑'] = ssim(pil_pred, pil_true).item()
            M.loc[(file), 'LPIPS ↑'] = lpips(pil_pred, pil_true).item()
            M.loc[(file), 'DeltaE ↓'] = np.mean(deltaE_cie76(color.rgb2lab(y_pred), color.rgb2lab(y_true)))

            print(M.loc[(file)])

    M.sort_index(inplace=True)
    print(M)

    createDir(f"{out_dir}/metrics/")
    # pickle.dump(M, open(f"{out_dir}/metrics_{dsName}/metrics.pkl", "wb"))
    M.to_csv(f"{out_dir}/metrics/metrics_newV2_l0.csv")

    # average the metrics of all the images
    M2 = M.mean(axis=0, numeric_only=True)
    M2 = pd.DataFrame(M2).T
    M2.to_csv(f"{out_dir}/metrics/metrics_new_l0.csv")
    return M

def createDir(name):
    os.makedirs(name, exist_ok=True)

if __name__ == "__main__":

    opt = {}

    print("\n\n------------------------------------------------")
    # opt["low_dir"] = "/ghome/mpilligua/DocumentConversion/Data/OurTest/gt_all"
    # opt["high_dir"] = "/ghome/mpilligua/DocumentConversion/Data/OurTest/l0"
    # opt["pred_dir"] = "/ghome/mpilligua/DocumentConversion/Results_paper/Inferencia/Restormer-3Ch/OurTest/Real_Denoising/"
    # opt["low_dir"] = "/ghome/mpilligua/DocumentConversion/Synthetic_DS_v5/test/gt_all_subset"
    # opt["high_dir"] = "/ghome/mpilligua/DocumentConversion/Synthetic_DS_v5/test/gt_l0_subset"
    # opt["pred_dir"] = "/ghome/mpilligua/DocumentConversion/Results_paper/Inferencia/Restormer-6Ch_lossChanged/Millor_l1/Test_Synth/Real_Denoising"

    opt["low_dir"] = "/ghome/mpilligua/DocumentConversion/Data/NewTest/ALL"
    opt["high_dir"] = "/ghome/mpilligua/DocumentConversion/Data/NewTest/L0"

    # opt["pred_dir"] = "/ghome/mpilligua/DocumentConversion/Results_paper/Inferencia/Restormer-3Ch/NewTest_3300/Real_Denoising/"
    # opt["pred_dir"] = "/ghome/mpilligua/DocumentConversion/Results_paper/Inferencia/Restormer-3Ch/NewTest_6000/Real_Denoising/"
    # opt["pred_dir"] = "/ghome/mpilligua/DocumentConversion/Results_paper/Inferencia/Restormer-6Ch/Millor_l0/NewTest/Real_Denoising/"
    # opt["pred_dir"] = "/ghome/mpilligua/DocumentConversion/Results_paper/Inferencia/Restormer-6Ch/Millor_l1/NewTest/Real_Denoising/"
    opt["pred_dir"] = "/ghome/mpilligua/DocumentConversion/Results_paper/Inferencia/Restormer-6Ch/Millor_l1/NewTest/Real_Denoising"

    # opt["pred_dir"] = "/ghome/mpilligua/DocumentConversion/Results_paper/Inferencia/Restormer-6Ch_lossChanged/Millor_l1/NewTest/Real_Denoising/"
    # opt["pred_dir"] = "/ghome/mpilligua/DocumentConversion/Results_paper/Inferencia/MirNet-3Ch/NewTest/"
    # opt["pred_dir"] = "/ghome/mpilligua/DocumentConversion/Results_paper/Inferencia/MirNet-6Ch/Millor_l0/NewTest/"
    # opt["pred_dir"] = "/ghome/mpilligua/DocumentConversion/Results_paper/Inferencia/MirNet-6Ch/Millor_l1/NewTest/"

    opt["save_dir"] = opt["pred_dir"]

    compute_metrics(opt)
    
    
