import os
import matplotlib.pyplot as plt
from PIL import Image


# Set the default pil image size to None
Image.MAX_IMAGE_PIXELS = None


FINAL_TEST_SET = ['Acta de nacimiento electrónica_0',
                  'Acta de nacimiento del estado de Veracruz_0',
                  'Constancia de no antecedentes penales federales_1',
                  'Estado de cuenta Banorte_Censurado_4',
                  'Estado de cuenta HSBC_Censurado_4',
                  'Estado de cuenta HSBC_Censurado_5',
                  'Título de licenciatura de la UV con apostilla_0',
                  'Título electrónico de licenciatura_0',
                  'Título electrónico escaneado_0',
                  'Título Universidad Veracruzana legalizado con apostilla_1',
                  'Acta de nacimiento estado de Guerrero_0',
                  'Acta de nacimiento estado de Guerrero_1',
                  'Boleta de calificaciones_0',
                  'CALP910804MCSLRT04_0',
                  'Certificado de estudios de bachillerato con apostilla_1',
                  'Certificado de registro de obra',
                  'Estado de cuenta Banorte_Censurado_2'
                  ]


# folder_input = '/ghome/mpilligua/basura/DIBCOSETS/Sample documents - JPG'
folder_input = "/ghome/mpilligua/DocumentConversion/Data/NewTest/ALL"
folder_L0 = "/ghome/mpilligua/DocumentConversion/Results_paper/Inferencia/Restormer-6Ch/Millor_l0/NewTest/Real_Denoising/"
folder_L1 = "/ghome/mpilligua/DocumentConversion/Results_paper/Inferencia/Restormer-6Ch/Millor_l0/NewTest/Real_Denoising/"
folder_GT_L0 = "/ghome/mpilligua/DocumentConversion/Data/NewTest/L0"
folder_GT_L1 = "/ghome/mpilligua/DocumentConversion/Data/NewTest/L1"

out_folder = '/ghome/mpilligua/DocumentConversion/Results_paper/Inferencia/Comparison_Restormer_Layers'
os.makedirs(out_folder, exist_ok=True)

import os
import matplotlib.pyplot as plt

reversed_list = os.listdir(folder_input)[::-1]
for file in reversed_list:
    
    name_pngl0 = file.replace('.png', '_l0.png')
    name_pngl1 = file.replace('.png', '_l1.png')
    
    img_folder_L0 = os.path.join(folder_L0, name_pngl0)
    img_folder_L1 = os.path.join(folder_L1, name_pngl1)
    img_folder_GT_L0 = os.path.join(folder_GT_L0, file)
    img_folder_GT_L1 = os.path.join(folder_GT_L1, file)
    img_folder_input = os.path.join(folder_input, file)
    
    fig, axs = plt.subplots(2, 3, figsize=(30, 20))

    input_image = plt.imread(img_folder_input)
    img_L0 = plt.imread(img_folder_L0)
    img_L1 = plt.imread(img_folder_L1)
    img_GT_L0 = plt.imread(img_folder_GT_L0)
    img_GT_L1 = plt.imread(img_folder_GT_L1)

    axs[0, 0].imshow(input_image)
    axs[0, 0].set_title('Input Image')
    axs[0, 0].axis('off')

    axs[1, 0].imshow(input_image)
    axs[1, 0].set_title('Input Image')
    axs[1, 0].axis('off')

    axs[0, 1].imshow(img_L0)
    axs[0, 1].set_title('Restored L0')
    axs[0, 1].axis('off')

    axs[0, 2].imshow(img_GT_L0)
    axs[0, 2].set_title('GT L0')
    axs[0, 2].axis('off')
    
    axs[1, 1].imshow(img_L1)
    axs[1, 1].set_title('Restored L1')
    axs[1, 1].axis('off')

    axs[1, 2].imshow(img_GT_L1)
    axs[1, 2].set_title('GT L1')
    axs[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, file))
    plt.close()

    
    