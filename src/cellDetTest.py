from hed_class import hed
import os


os.environ["CUDA_VISIBLE_DEVICES"]="0"

if __name__ == '__main__':
    
    hedModel = hed(trainImSize=[304, 304], testImSize=[304, 304])
    
    hedModel.modelPred(modelFilePath = '../models/cellDet_hed_model/model.ckpt-240', testImgFolder = '../data/cell_det_data/test/3', 
                  testOneFile = False, testImgPath= '../data/cell_det_data/test/img_15_5.png', resultFolder = '../data/cell_det_data/res/3/', 
                  trainDataMean = 188.9, resizeto=[800, 800], grayImg = 0, rgbImg= 1)
    
    
    
    #testImgPath = '../data/cell_det_data/test/TCGA-A8-A08I41653_50243_08.jpg')
    