from glob import glob
import numpy as np
from skimage import measure
import visdom
import os
from natsort import natsorted
import cv2

from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score

vis = visdom.Visdom()


def measureCellSize(contours, img):
    h,w,chan = img.shape
    areas = []
    for cid in contours:
        xy_min = cid.min(0)
        x_min, y_min = xy_min[0,0], xy_min[0,1]
        
        xy_max = cid.max(0)
        x_max, y_max = xy_max[0,0], xy_max[0,1] 
        
        workPadSize = max(x_max - x_min, y_max - y_min)
        
        workPad = np.zeros((workPadSize, workPadSize), dtype = np.uint8)
        workPad_ = np.zeros((workPadSize+2, workPadSize+2), dtype = np.uint8)
        contourCoord = cid - np.asarray([x_min, y_min])
        mask = cv2.drawContours(workPad, [contourCoord], -1, (1, 1, 1), 1)
        contourCenter = contourCoord.mean(0)
        contourCenter = np.round(contourCenter).astype(np.uint8)
        cv2.floodFill(workPad, workPad_, (contourCenter[0][0], contourCenter[0][1]), 255)
        workPad[workPad<255] = 0
        workPad[workPad == 255] = 1
        labels, num_conn = measure.label(workPad, neighbors=4, background=0, return_num=True, connectivity=1)
        regProps = measure.regionprops(labels)
        area = regProps[0].area
        areas.append(area)
    
    areas = np.asarray(areas, np.float32)
    
    area_var = areas.var()
    area_mean = areas.mean()
    
    return area_mean, area_var


segResFolder = '/media/hai/data/hai/cellDetSeg_HED_VAE_tf/data/cell_seg_data/res/1/'
imgFolder = '/media/hai/data/hai/cellDetSeg_HED_VAE_tf/data/cell_det_data/test/1/'

segResFiles = glob(segResFolder + '*.npy')
imgFiles = glob(imgFolder + '*.png')

segResFiles = natsorted(segResFiles)
imgFiles = natsorted(imgFiles)

imgLevelCellSize_mean = []
imgLevelCellSize_var = []
grade1_feat = []
for imgFile, segResFile in zip(imgFiles, segResFiles):
    imgPath, imgName = os.path.split(imgFile)
    imgN, ext = imgName.split('.')
    resPath, resName = os.path.split(segResFile)
    resN, ext = resName.split('.')
    if imgN == resN:
        oneImgContours = np.load(segResFile)
        im = cv2.imread(imgFile)
        
        cellSize_mean, cellSize_var = measureCellSize(oneImgContours, im)
        imgLevelCellSize_mean.append(cellSize_mean)
        imgLevelCellSize_var.append(cellSize_var)
        grade1_feat.append([cellSize_mean, cellSize_var])

imgLevelCellSize_mean = np.asarray(imgLevelCellSize_mean)
imgLevelCellSize_var = np.asarray(imgLevelCellSize_var)

bins_for_mean = [200,400,600, 800, 1000,1200, 1400, 1600]
bins_for_var = range(10000,400000, 20000)

print("histogram of cell size mean: ")
print(np.histogram(imgLevelCellSize_mean, bins=bins_for_mean))

print("\n")
print("histogram of cell size var: ")
print(np.histogram(imgLevelCellSize_var, bins=bins_for_var))


segResFolder = '/media/hai/data/hai/cellDetSeg_HED_VAE_tf/data/cell_seg_data/res/3/'
imgFolder = '/media/hai/data/hai/cellDetSeg_HED_VAE_tf/data/cell_det_data/test/3/'

segResFiles = glob(segResFolder + '*.npy')
imgFiles = glob(imgFolder + '*.png')

segResFiles = natsorted(segResFiles)
imgFiles = natsorted(imgFiles)

imgLevelCellSize_mean = []
imgLevelCellSize_var = []
grade3_feat = []
for imgFile, segResFile in zip(imgFiles, segResFiles):
    imgPath, imgName = os.path.split(imgFile)
    imgN, ext = imgName.split('.')
    resPath, resName = os.path.split(segResFile)
    resN, ext = resName.split('.')
    if imgN == resN:
        oneImgContours = np.load(segResFile)
        im = cv2.imread(imgFile)
        
        cellSize_mean, cellSize_var = measureCellSize(oneImgContours, im)
        imgLevelCellSize_mean.append(cellSize_mean)
        imgLevelCellSize_var.append(cellSize_var)
        grade3_feat.append([cellSize_mean, cellSize_var])
        

#################################################################

grade1_feat = np.asarray(grade1_feat, dtype = np.float32)
grade1_target = np.zeros((len(grade1_feat),), dtype = np.float32)

grade3_feat = np.asarray(grade3_feat, dtype = np.float32)
grade3_target = np.ones((len(grade3_feat), ), dtype = np.float32)


train_feat = np.concatenate((grade1_feat, grade3_feat), axis = 0)
train_target = np.concatenate((grade1_target, grade3_target), axis=0) 

#test_feat = np.concatenate((grade1_feat[30:,:], grade3_feat[20:,:]), axis = 0)
#test_target = np.concatenate((grade1_target[30:], grade3_target[20:]), axis=0)


clf = svm.SVC(kernel='linear', C=1) #.fit(train_feat, train_target)
#clf.score(test_feat, test_target)
scores = cross_val_score(clf, train_feat, train_target, cv=2)
print("cross validation: ")
print(scores.mean())
print(scores.std())