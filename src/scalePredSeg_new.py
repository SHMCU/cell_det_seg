import numpy as np
import glob
import os
from PIL import Image
import cv2
from scipy import misc
from scalePredNet import scalePredNet
#import cPickle
import pickle
import gzip

'''import theano
import theano.tensor as T

from VAE_cell import VAE_cell'''

import matplotlib
#from backports.configparser import Interpolation
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from pylab import *
import pylab
from conVAE import createConvVAE
import skimage.transform
from skimage import measure
import scipy.io as sio
import visdom

vis = visdom.Visdom()

def drawBinSegMask(contours, img):
    h, w, chan = img.shape
    mask = np.zeros((h,w), dtype = np.uint8)
    mask_ = np.zeros((h+2, w+2), dtype = np.uint8)
    for cid in contours:
        mask = cv2.drawContours(mask, [cid], -1, (1, 1, 1), 1)
        contourCenter = cid.mean(0)
        contourCenter = np.round(contourCenter).astype(np.uint32)
        cv2.floodFill(mask, mask_, (contourCenter[0][0], contourCenter[0][1]), 255)
    
    mask= cv2.copyMakeBorder(mask,1,1,1,1,cv2.BORDER_CONSTANT,value=0)
    mask = mask[1:-1, 1:-1]
    
    return mask

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


def getPatch(patCenterY, patCenterX, patSize, origImg):
    row, col= origImg.shape
    patStartY = patCenterY-patSize
    patEndY = patCenterY+patSize
    patStartX = patCenterX-patSize
    patEndX = patCenterX+patSize
    outOfRng = 0

    if patStartX >= 1 and patEndX <= row and patStartY >= 1 and patEndY <= col:
        onePatch = origImg[patStartX: patEndX, patStartY: patEndY]
        onePatch = misc.imresize(onePatch, (40, 40), interp = 'bilinear')
        onePatch = np.true_divide(onePatch, onePatch.max())
        onePatch = np.asarray(onePatch, dtype=np.float32)
    else:
        outOfRng = 1
        onePatch = 0
    
    return onePatch, outOfRng

def getPatch_colorImg(patCenterY, patCenterX, patSize, origImg):
    row, col, chan= origImg.shape
    patStartY = patCenterY-patSize
    patEndY = patCenterY+patSize
    patStartX = patCenterX-patSize
    patEndX = patCenterX+patSize
    outOfRng = 0

    if patStartX >= 1 and patEndX <= row and patStartY >= 1 and patEndY <= col:
        onePatch = origImg[patStartX: patEndX, patStartY: patEndY, :]
        origColorPatch = skimage.transform.resize(onePatch, (64, 64, 3), preserve_range=True)
        onePatch = skimage.transform.resize(onePatch, (96, 96, 3), preserve_range=True)
        onePatch = np.float32(onePatch.copy())
        onePatch[:,:,0] = (onePatch[:,:,0] - 169)/32.5
        onePatch[:,:,1] = (onePatch[:,:,1] - 113)/45.9
        onePatch[:,:,2] = (onePatch[:,:,2] - 164)/43.75
        
        #onePatch = misc.imresize(onePatch, (96, 96), interp = 'bilinear')
        
        
        #onePatch = np.true_divide(onePatch, onePatch.max())
        #onePatch = np.asarray(onePatch, dtype=np.float32)
    else:
        outOfRng = 1
        onePatch = 0
        origColorPatch = 0
    
    return onePatch, outOfRng, origColorPatch



def cropPyramidPatchs(seeds = None, img = None, scales = [12, 15, 18, 21, 25,30]):
    
    row, col, chan = np.shape(img)
    oneImgPatches = np.empty(shape=(len(seeds), len(scales), 64, 64, 3), dtype = np.float32)
    
    for ind_s, det in enumerate(seeds):
        oneDetPatches = np.empty(shape = (len(scales), 64, 64, 3), dtype = np.float32)
        
        for ind_scl, scale in enumerate(scales):
            minY = int(det[0]-scale)
            maxY = int(det[0]+scale)
            minX = int(det[1]-scale)
            maxX = int(det[1]+scale)
            if maxY < row and minY > -1 and maxX < col and minX > -1:
                onePatch = img[minY:maxY, minX:maxX, :]
                onePatch = misc.imresize(onePatch, (64, 64), interp = 'bilinear')
                onePatch = np.true_divide(onePatch, onePatch.max())
                #onePatch = np.repeat(onePatch[:, :, np.newaxis], 3, axis=2)
                onePatch[:,:,0] -= 0.677 # img_mean_b
                onePatch[:,:,1] -= 0.395 # img_mean_g
                onePatch[:,:,2] -= 0.687 # img_mean_r
                oneDetPatches[ind_scl, :, :, :] = onePatch 
        oneImgPatches[ind_s] = oneDetPatches
    return oneImgPatches


def predScale_Seg(imgPath = '../data/cell_det_data/test/3/', detResPath = '../data/cell_det_data/res/3/', segResPath = '../data/cell_seg_data/res/3/', bdxScales = [12, 15, 18, 21, 25, 30, 35, 39], resizeto=[1008, 1008]):
    detFiles = glob.glob(detResPath + '*.npy')
    detFiles = sorted(detFiles)
    imgFiles = glob.glob(imgPath + '*.png')
    imgFiles = sorted(imgFiles)
    
    # create scale prediction model
    scalePredModel = scalePredNet()
    scale_Pred_Model = scalePredModel.createNet(imgShape=[64, 64])
    vae = createConvVAE()
    vae.load_weights(filepath='../models/convVAE_model/weights.00-630.35.hdf5', by_name = True)
    
    # create segmentation model
    '''f = gzip.open('basic0.pkl.gz', 'rb')
    (x_train, t_train), (x_valid, t_valid), (x_test, t_test) = cPickle.load(f)
    f.close()
    path = "../models/cellSeg_vae_model/"
    print "instantiating model"
    continuous = False
    hu_encoder = 800
    hu_decoder = 800
    n_latent = 48
    model = VAE_cell(continuous, hu_encoder, hu_decoder, n_latent, x_train, x_valid)
    model.load_parameters(path)
    
    x_ = T.vector('x_')
    x_clean = T.vector('x_clean')
    mu, log_sigma = model.encoder(x_)
    z = model.sampler(mu, log_sigma)
    reconstructed_x, logpxz = model.decoder(x_,z, x_clean)
    reconFunc = theano.function([x_], [reconstructed_x])'''
    
    # start the processing loop over the images:
    imgLevelCellSize_mean = []
    imgLevelCellSize_var = []
    for imgFile, detFile in zip(imgFiles, detFiles):
        _, imgName = os.path.split(imgFile)
        _, detName = os.path.split(detFile)
        if imgName[:-4] == detName[:-4]:
            img = cv2.imread(imgFile)
            origImg = np.copy(img)
            
            #imWdt, imHt = img.size()
            #img = cv2.resize(img, (resizeto[0], resizeto[1]), interpolation=cv2.INTER_LINEAR)
            img = np.asarray(img, dtype=np.float32)
            
            img_gray = np.dot(img[...,:3], [0.299, 0.587, 0.114])
            img_gray = np.asarray(img_gray, dtype = np.float32)
            
            seeds = np.load(detFile)
            
            oneImgPatches = cropPyramidPatchs(seeds, img)
            #oneImgPatches = oneImgPatches.transpose(1,2,3,0)
            noSeeds = np.shape(oneImgPatches)[0]
            noScales = np.shape(oneImgPatches)[1]
            oneImgPatches = np.reshape(oneImgPatches, newshape=(noSeeds*noScales, 64, 64, 3))
            
            scales = scale_Pred_Model.predict(oneImgPatches, 256, verbose=1)
            pred_scales = scales[:, 1]
            indx1 = pred_scales > 0.5
            indx2 = pred_scales <= 0.5
            pred_scales[indx1] = 1
            pred_scales[indx2] = 0
            pred_scales = np.reshape(pred_scales, newshape=(noSeeds, noScales))
            
            bdxSize = np.empty(shape=(noSeeds), dtype = np.int16)
            for seedInd, oneCellScales in enumerate(pred_scales):
                temp_ind = np.where(oneCellScales == 1)
                
                if temp_ind[0] != []:
                    bdxSize[seedInd] = bdxScales[temp_ind[0][0]]
                else:
                    bdxSize[seedInd] = bdxScales[-1]
            
            # visualize the scale prediction results in image by plotting the bounding boxes
            '''for boundingboxsize, seed in zip(bdxSize, seeds):
                x = np.int16(seed[1])
                y = np.int16(seed[0])                
                cv2.rectangle(img, (x-boundingboxsize, y-boundingboxsize), (x+boundingboxsize, y+boundingboxsize), (0, 255, 0), thickness=2)    
            cv2.imwrite(imgName[:-3] + 'jpg', img)'''
            
            # start segmentation
            localCounter = 1
            origColorPatches = []
            testPatch = []
            patchLoc = []
            origPatchSize = []
            img_gray_ = img_gray/img_gray.max() 
            for oneDet, patSize in zip(seeds, bdxSize):
                oneSeed = np.int16(oneDet)
                
                #onePatch_, flag = getPatch(oneSeed[1], oneSeed[0], patSize, img_gray_)
                onePatch_, flag, colorPatch = getPatch_colorImg(oneSeed[1], oneSeed[0], patSize, origImg)
                
                if flag !=1:
                    #onePatch__ = np
                    #plt.figure()
                    #plt.imshow(onePatch_)
                    #plt.show()
                    #onePatch_vector = onePatch_.flatten()
                    origColorPatches.append(colorPatch)
                    testPatch.append(onePatch_)
                    patchLoc.append(oneSeed)
                    origPatchSize.append(patSize)
                    localCounter = localCounter + 1
        
            testPatch = np.array(testPatch)
            
            pred_results = vae.predict(testPatch)
            
            outputContours = []
            localCounter = 1
            for i in range(len(pred_results[2])):
                origSize = origPatchSize[i]
                res = pred_results[2][i,:,:,0]
                res = skimage.transform.resize(res, (origSize*2, origSize*2), preserve_range=True)
                
                res[res>0.1] = 255
                #res = (res - res.min())/((res.max() - res.min())/255)
                res = np.uint8(res)
                ret, res = cv2.threshold(res, 50, 255, cv2.THRESH_BINARY)
                origSize = res.shape[0]
                mask = np.zeros((origSize+2,origSize+2), np.uint8)
                cv2.floodFill(res, mask, (origSize//2, origSize//2), 255)
                res= cv2.copyMakeBorder(res,1,1,1,1,cv2.BORDER_CONSTANT,value=0)
                kernel = np.ones((2,2),np.uint8)
                res = cv2.erode(res, kernel,iterations = 1)
                
                res = res[1:-1, 1:-1]
                contours, hierarchy = cv2.findContours(res, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                outputContours.append(contours[0])
                localCounter = localCounter + 1
            
            """encode_ = theano.function([x_], z)
            decode_ = theano.function([z], reconstructed_x)"""
            
            '''outputContours = [] 
            localCounter = 1
            for testSample, origSize in zip(testPatch, origPatchSize):
                    
                reconImg1 = reconFunc(testSample)
                #reconImg2 = reconFunc(testSample)
                #reconImg3 = reconFunc(testSample)
                reconImg_1 = np.asanyarray(reconImg1, dtype=np.float32)
                reconImg_1 *= 255.0/reconImg_1.max() 
                reconImg_1 = reconImg_1.reshape(40, 40)
                
                reconImg_1_bin = np.uint8(reconImg_1)
                reconImg_1_bin = misc.imresize(reconImg_1_bin, size=(origSize*2, origSize*2), interp='bilinear')
                
                ret, reconImg_1_bin = cv2.threshold(reconImg_1_bin, 50, 255, cv2.THRESH_BINARY)
                testReconImg_1_bin = reconImg_1_bin
                mask = np.zeros((origSize*2+2,origSize*2+2), np.uint8)
                cv2.floodFill(reconImg_1_bin, mask, (origSize, origSize), 255);
                reconImg_1_bin= cv2.copyMakeBorder(reconImg_1_bin,1,1,1,1,cv2.BORDER_CONSTANT,value=0)
                kernel = np.ones((2,2),np.uint8)
                reconImg_1_bin_ = cv2.erode(reconImg_1_bin,kernel,iterations = 2)
                
                reconImg_1_bin_ = reconImg_1_bin_[1:origSize*2+1, 1:origSize*2+1]
                contours, hierarchy = cv2.findContours(reconImg_1_bin_, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                outputContours.append(contours[0])
                
                cv2.drawContours(reconImg_1_bin, contours, -1, (255,255,0), 1)
                cv2.imshow("Keypoints", reconImg_1_bin)
                                    
                pylab.figure()
                pylab.gray()
                pylab.imshow(np.reshape(testSample, (40, 40)), interpolation='nearest')
                
                pylab.figure()
                pylab.gray()
                pylab.imshow(reconImg_1_bin, interpolation='nearest');
                pylab.figure()
                pylab.gray()
                pylab.imshow(np.reshape(reconImg_1, (40, 40)), interpolation='nearest')
                pylab.figure()
                pylab.gray()
                pylab.imshow(testReconImg_1_bin, interpolation='nearest')
                pylab.show()
                
                localCounter = localCounter + 1'''
            
            img_ = img
            imht, imwd, channel = np.shape(img_)
            imgDim = max(imwd, imht)
            img_ = np.array(img_, dtype=np.float32)
            img_cv = img_[:, :, ::-1].copy()
            
            contours = list()
            localCounter = 1
            finalContours = []
            for indx in range(len(outputContours)):
                oneContour = outputContours[indx]
                if oneContour.shape[0] > 25:
                    oneSeed = patchLoc[indx]
                    oneContour[:,0,0] = oneContour[:,0,0]+oneSeed[1]-origPatchSize[indx]
                    oneContour[:,0,1] = oneContour[:,0,1]+oneSeed[0]-origPatchSize[indx]
                    finalContours.append(oneContour) 
                    #print oneContour_.shape 
                    #pylab.plot((oneContour[:,0, 0]+oneSeed[1]-20)/imgResizeRatio, (oneContour[:,0, 1]+oneSeed[0]-20)/imgResizeRatio, 'g', linewidth=1/(imgDim*0.001))
            #plt.axis('off')
            cv2.drawContours(img_cv, finalContours, -1, (0, 255, 0), 2)
            cv2.imwrite(segResPath + imgName[:-3] + 'jpg', img_cv)     
            np.save(segResPath + imgName[:-4]+ '_contour' + '.npy', finalContours)
            sio.savemat(segResPath + imgName[:-4]+ '_contour' + '.mat', {'contours': finalContours})
            
            mask = drawBinSegMask(finalContours, img_cv)
            #cv2.imwrite(segResPath + imgName[:-4] + '_seg' + '.png', mask)
            
            cellSize_mean, cellSize_var = measureCellSize(finalContours, img_cv)
            imgLevelCellSize_mean.append(cellSize_mean)
            imgLevelCellSize_var.append(cellSize_var)
            
            origColorPatches = np.asarray(origColorPatches, dtype=np.float32)
            #np.save('/media/hai/7840-BBFF/cellPatches/' + imgName[:-4] + '_patch' + '.npy', origColorPatches)
            
        else:
            print('The image and the detection results do not match !')
    
    return imgLevelCellSize_mean, imgLevelCellSize_var
            
if __name__ == '__main__':
    cellSizeMean, cellSizeVar = predScale_Seg()
    print(cellSizeMean)
    print(cellSizeVar)