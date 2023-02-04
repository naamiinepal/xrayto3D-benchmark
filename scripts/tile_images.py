from pathlib import Path
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

INPUT_DIR = r'unet-femur-output'
OUTPUT_DIR = r'.'

files = sorted(list(Path(INPUT_DIR).rglob('*.png')))
SINGLE_IMG_SZ = 300
GRID_COL = 10
GRID_ROW = 6
MONTAGE = Image.new('RGB',(SINGLE_IMG_SZ*GRID_COL,SINGLE_IMG_SZ*GRID_ROW))

for c, i in enumerate(range(0,len(files),4)):
    # if c > (GRID_COL//2  * GRID_ROW//2 ):
    #     break
    row, col  = (c*2// GRID_COL)*2 , (c*2) % GRID_COL
    print(f'c {c} i {i} row {row} col {col}')

    gt_coronal, gt_sagittal, pred_coronal, pred_sagittal = files[i+0],files[i+1],files[i+2],files[i+3]
    print(gt_coronal,gt_sagittal,pred_coronal,pred_sagittal)

    gt_coronal = Image.open(gt_coronal)
    gt_sagittal = Image.open(gt_sagittal)
    pred_coronal = Image.open(pred_coronal)
    pred_sagittal = Image.open(pred_sagittal)

    gt_coronal_index = [0,0]
    gt_sagittal_index = [0,1]
    pred_coronal_index = [1,0]
    pred_sagittal_index = [1,1]
    for c,(index,img) in enumerate(zip((gt_coronal_index,gt_sagittal_index,pred_coronal_index,pred_sagittal_index),
    (gt_coronal,gt_sagittal,pred_coronal,pred_sagittal))):
        coord_i, coord_j = index
        coord_i = (row+coord_i) * SINGLE_IMG_SZ
        coord_j = (col+coord_j) * SINGLE_IMG_SZ
        if c < 2: # gt
            bg = Image.new('RGBA', (SINGLE_IMG_SZ,SINGLE_IMG_SZ), (255,255,255,50))
            img = Image.composite(bg,img,bg)
        MONTAGE.paste(img,(coord_i,coord_j))

MONTAGE.save(Path(OUTPUT_DIR)/'unet-femur-montage.pdf')
