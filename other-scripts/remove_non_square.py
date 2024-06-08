# iterate over the images in the folder and remove the ones that are not 512 x 512 

import os
import cv2
from glob import glob
from natsort import natsorted
from tqdm import tqdm

src = '/ghome/mpilligua/DocumentConversion/data/DIBCOSETS/training_2009-2019_crops'

l0 = os.path.join(src, 'all')

files = natsorted(glob(os.path.join(l0, '*.png')))

total = 0
for file_ in tqdm(files):
    img = cv2.imread(file_)
    if img.shape[0] != 512 or img.shape[1] != 512:
        os.remove(file_)
        print(f"Removed {file_} with shape {img.shape}")
        
print(total)