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
folder_input = '/ghome/mpilligua/DocumentConversion/Data/NewTest/ALL'
folder_6channel = '/ghome/mpilligua/DocumentConversion/Results_paper/Inferencia/Restormer-6Ch/Millor_l0/NewTest/Real_Denoising'
folder_3channel_6000 = '/ghome/mpilligua/DocumentConversion/Results_paper/Inferencia/Restormer-3Ch/NewTest_6000/Real_Denoising'
folder_3channel_3000 = '/ghome/mpilligua/DocumentConversion/Results_paper/Inferencia/Restormer-3Ch/NewTest_3300/Real_Denoising'


out_folder = '/ghome/mpilligua/DocumentConversion/Results_paper/Inferencia/Comparrison_Restormer_NewTest'

import os
import matplotlib.pyplot as plt

reversed_list = os.listdir(folder_input)[::-1]
for file in reversed_list:
    file_path = os.path.join(folder_input, file)
    
    name_png = file.replace('jpg', 'png')
    name_pngl0 = name_png.replace('.png', '_l0.png')
    
    img_folder_6channel_l0 = os.path.join(folder_6channel, name_pngl0)
    img_folder_3channel_6000 = os.path.join(folder_3channel_6000, name_png)
    img_folder_3channel_3000 = os.path.join(folder_3channel_3000, name_png)
    
    
    #if os.path.exists(os.path.join(out_folder, name_png)):
    #    continue
    input_image = plt.imread(file_path)
    img_6channel_l0 = plt.imread(img_folder_6channel_l0)
    img_3channel_6000 = plt.imread(img_folder_3channel_6000)
    img_3channel_3000 = plt.imread(img_folder_3channel_3000)
    
    # Put the images in a grid
    fig, axs = plt.subplots(1, 4, figsize=(20, 20))
    
    axs[0].imshow(input_image)
    axs[0].set_title('Input image')
    axs[0].axis('off')  # Hide axis
    
    axs[1].imshow(img_6channel_l0)
    axs[1].set_title('6 channel l0')
    axs[1].axis('off')  # Hide axis
    
    axs[2].imshow(img_3channel_6000)
    axs[2].set_title('3 channel 6000')
    axs[2].axis('off')  # Hide axis
    
    axs[3].imshow(img_3channel_3000)
    axs[3].set_title('3 channel 3000')
    axs[3].axis('off')  # Hide axis
    
    # Save the image
    out_folder_file = os.path.join(out_folder, name_png)
    plt.savefig(out_folder_file, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

    
    