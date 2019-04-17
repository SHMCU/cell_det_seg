import numpy as np
from glob import glob
import os
import visdom
import cv2

vis = visdom.Visdom()

cellPatchFolder = '/media/hai/7840-BBFF/cellPatches_1/' 

outputFolder = '/media/hai/7840-BBFF/cellPatches_imgs1/'

cellPatchFiles = glob(cellPatchFolder + '*.npy')

for fid in cellPatchFiles:
    filePath, fileName = os.path.split(fid)
    fileName, ext = fileName.split('.')
    
    cellPatchArray = np.load(fid)
    no_patches = len(cellPatchArray)
    for i in range(no_patches):
        if i % 3 ==0:
            onePatchImg = cellPatchArray[i,:,:,:]
            onePatchImg_out = onePatchImg.astype(np.uint8)
            imgName = fileName + '_' +str(i)
            cv2.imwrite(outputFolder + imgName + '.png', onePatchImg_out)
        else:
            continue