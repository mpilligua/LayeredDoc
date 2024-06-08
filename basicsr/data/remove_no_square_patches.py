# dir to scan /hhome/ps2g07/document_analysis/github/Project_Synthesis2-/Datasets/Synthetic_DS_v3/train_crops

# check that all patches are square

import os
import cv2

count = 0
for root, dirs, files in os.walk("/hhome/ps2g07/document_analysis/github/Project_Synthesis2-/Datasets/Synthetic_DS_v3/train_crops"):
    for file in files:
        if file.endswith(".png"):
            path = os.path.join(root, file)
            img = cv2.imread(path)
            h, w, _ = img.shape
            if h != w:
                print(f"Image {file} is not square", img.shape)
                # os.remove(path)
                # print(f"Removed {path}")
                count += 1
            # else:
                # print(f"Image {path} is square")
                
print("total non square", count)