from src.hed_class import hed
import os
from src.scalePredSeg_new import *
import argparse

os.environ["CUDA_VISIBLE_DEVICES"]="0"


def main():
    parser = argparse.ArgumentParser(description="Nucleus detection and segmentation testing!")
    parser.add_argument("--detModel", default="./models/cellDet_hed_model/model.ckpt-240", help="path to HED detection model file", type=str)
    parser.add_argument("--patchSize", default=[304, 304], help="the patch size used to train the HED network", type=list)
    parser.add_argument("--testImgs", type=str, default="./data/cell_det_data/test/3/", help="path to testing images")
    parser.add_argument("--testOneFile", type=bool, default=False, help="Indicate to test one image or multiple images, if one image, provide the path to that image")
    parser.add_argument("--testImgPath", default= "./data/cell_det_data/test/img_15_5.png", type=str, help="Provide the path to one testing image")
    parser.add_argument("--detResultFolder", default = "./data/cell_det_data/res/3/", type=str, help="the folder to save the detection results")
    
    
    parser.add_argument("--scalePredModel", default= "./models/cellSizePred_vgg16_model/checkpoints/weights.239-0.10.hdf5", help="path to the segmentation model file", type=str)
    parser.add_argument("--segModel", default= "./models/convVAE_model/weights.00-630.35.hdf5", help="path to the segmentation model file", type=str)
    parser.add_argument("--segResultFolder", default = "./data/cell_seg_data/res/3/", type=str, help="the folder to save the segmentation results")
    parser.add_argument("--isGrayImg", type=int, default = 0)
    parser.add_argument("--isColorImg", type=int, default = 1)
    parser.add_argument("--bdxScales", type=list, default = [12, 15, 18, 21, 25, 30, 35, 39], help = "possible bounding box sizes for nucleus segmentation")
    parser.add_argument("--resizeTo", type=list, default = [1008, 1008], help = "resize the input image to this size")
    args = parser.parse_args()


    hedModel = hed(trainImSize=args.patchSize, testImSize=args.patchSize)
    hedModel.modelPred(modelFilePath = args.detModel, testImgFolder = args.testImgs, 
                  testOneFile = args.testOneFile, testImgPath= args.testImgPath, resultFolder = args.detResultFolder, 
                  trainDataMean = 188.9, resizeto=[800, 800], grayImg = args.isGrayImg, rgbImg= args.isColorImg)
    predScale_Seg(scalePredModelFile = args.scalePredModel, segModel=args.segModel, imgPath = args.testImgs, 
                  detResPath = args.detResultFolder, segResPath = args.segResultFolder, bdxScales = args.bdxScales, resizeto=args.resizeTo)
    
if __name__ == '__main__':
    main()
    

