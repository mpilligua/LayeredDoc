# import module
from pdf2image import convert_from_path
import os
 
# Store Pdf with convert_from_path function

path = '/ghome/mpilligua/DocumentConversion/TestFinal'

pdfs = ['lpcopy_20240530_140226.pdf',
        'lpcopy_20240530_140252.pdf']

new_path = '/ghome/mpilligua/DocumentConversion/TestFinal/'
for idx, pdf in enumerate(pdfs):
    path_pdf = path + '/' + pdf
    images = convert_from_path(path_pdf)

    for idx, result in enumerate(images):
        name_png = pdf.replace('.pdf', f'_{idx}.png')
        new_path_png = os.path.join(new_path, name_png)
        result.save(new_path_png)
