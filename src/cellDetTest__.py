from hed_class import hed
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"

if __name__ == '__main__':
    
    hedModel = hed(trainImSize=[1008, 1008], testImSize=[1008, 1008])
    '''
    hedModel.modelPred(modelFilePath = '../models/cellDet_hed_model/model.ckpt-165', testImgFolder = '../data/cell_det_data/test/', 
                  testOneFile = True, testImgPath = '../data/cell_det_data/test/img_15_5.png', 
                  resultFolder = '../data/cell_det_data/res/', trainDataMean = 188.9)
    '''
    hedModel.modelPred(modelFilePath = '../models/cellDet_hed_model/model.ckpt-165', testImgFolder = '../data/cell_det_data/test/bmp/', 
                  testOneFile = False, testImgPath = '../data/cell_det_data/test/img_15_5.png', 
                  resultFolder = '../data/cell_det_data/res_yun/', trainDataMean = 188.9)
    
    