from networkUtils import convLayer, convLayer_upsample, getPretrainedModel
import tensorflow as tf
import os
import numpy as np
import cv2
import glob
from docutils.nodes import target
from scipy import ndimage
from skimage.feature import peak_local_max
from sklearn.cluster import MeanShift
from PIL import Image
import matplotlib.pylab as pylab


class hed(object):
    def __init__(self, trainImSize, testImSize):
        self.trainImHt, self.trainImWdt = trainImSize
        self.testImHt, self.testImWdt = testImSize
    
    def createHed(self, train_or_test = True):
        if train_or_test == True:
            img = tf.placeholder(dtype = tf.float32, shape = (None, self.trainImHt, self.trainImWdt, 3), name= 'image')
        else:
            img = tf.placeholder(dtype = tf.float32, shape = (None, self.testImHt, self.testImWdt, 3), name= 'image')
        
        with tf.variable_scope('conv1_1') as scope:
            x = convLayer(inputFeat=img, name='1', kernel_shape=[3,3,3, 64], stride_shape=[1,1,1,1])
        with tf.variable_scope('conv1_2') as scope:
            x = convLayer(inputFeat=x, name='2', kernel_shape=[3,3, 64, 64], stride_shape=[1,1,1,1])
        with tf.variable_scope('branch1') as scope:
            b1 = convLayer_upsample(x, name='b1', kernel_shape=[1,1,64,1], stride_shape=[1,1,1,1], up_ratio=1, actFun='identity')
        pool1 = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
        
        
        #norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
        with tf.variable_scope('conv2_1') as scope:
            x = convLayer(inputFeat=pool1, name='1', kernel_shape=[3, 3, 64, 128], stride_shape=[1,1,1,1])
        with tf.variable_scope('conv2_2') as scope:
            x = convLayer(inputFeat=x, name='2', kernel_shape=[3, 3, 128, 128], stride_shape=[1,1,1,1])
        with tf.variable_scope('branch2') as scope:
            b2 = convLayer_upsample(x, name='b2', kernel_shape=[1,1,128,1], stride_shape=[1,1,1,1], up_ratio=2, actFun='identity')
        pool2 = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
        
        with tf.variable_scope('conv3_1') as scope:
            x = convLayer(inputFeat=pool2, name='1', kernel_shape=[3, 3, 128, 256], stride_shape=[1,1,1,1])
        with tf.variable_scope('conv3_2') as scope:
            x = convLayer(inputFeat=x, name='2', kernel_shape=[3, 3, 256, 256], stride_shape=[1,1,1,1])
        with tf.variable_scope('conv3_3') as scope:
            x = convLayer(inputFeat=x, name='3', kernel_shape=[3, 3, 256, 256], stride_shape=[1,1,1,1])
        with tf.variable_scope('branch3') as scope:
            b3 = convLayer_upsample(x, name='b3', kernel_shape=[1, 1, 256, 1], stride_shape=[1,1,1,1], up_ratio=4, actFun='identity')
        pool3 = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
        
        with tf.variable_scope('conv4_1') as scope:
            x = convLayer(inputFeat=pool3, name='1', kernel_shape=[3, 3, 256, 512], stride_shape=[1,1,1,1])
        with tf.variable_scope('conv4_2') as scope:
            x = convLayer(inputFeat=x, name='2', kernel_shape=[3, 3, 512, 512], stride_shape=[1,1,1,1])
        with tf.variable_scope('conv4_3') as scope:
            x = convLayer(inputFeat=x, name='3', kernel_shape=[3, 3, 512, 512], stride_shape=[1,1,1,1])
        with tf.variable_scope('branch4') as scope:
            b4 = convLayer_upsample(x, name='b4', kernel_shape=[1, 1, 512, 1], stride_shape=[1,1,1,1], up_ratio=8, actFun='identity')
        pool4 = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')
        
        with tf.variable_scope('conv5_1') as scope:
            x = convLayer(inputFeat=pool4, name='1', kernel_shape=[3, 3, 512, 512], stride_shape=[1,1,1,1])
        with tf.variable_scope('conv5_2') as scope:
            x = convLayer(inputFeat=x, name='2', kernel_shape=[3, 3, 512, 512], stride_shape=[1,1,1,1])
        with tf.variable_scope('conv5_3') as scope:
            x = convLayer(inputFeat=x, name='3', kernel_shape=[3, 3, 512, 512], stride_shape=[1,1,1,1])
        with tf.variable_scope('branch5') as scope:
            b5 = convLayer_upsample(x, name='b5', kernel_shape=[1, 1, 512, 1], stride_shape=[1,1,1,1], up_ratio=16, actFun='identity')
        
        branchComb = tf.concat([b1,b2,b3,b4,b5], axis=3, name='branchComb')
        with tf.variable_scope('convfcweight') as scope:
            predEdgeMap = convLayer(inputFeat=branchComb, name='1', kernel_shape=[1, 1, 5, 1], stride_shape=[1,1,1,1], actFun='identity')
        
        return b1, b2, b3, b4, b5, predEdgeMap
    
    def class_balanced_sigmoid_cross_entropy(self, logits, edgeMapLabel, name='cross_entropy_loss'):
        
        y = tf.cast(edgeMapLabel, tf.float32)
        
        count_neg = tf.reduce_sum(1. - y)
        count_pos = tf.reduce_sum(y)
        beta = count_neg / (count_neg + count_pos)
    
        pos_weight = beta / (1 - beta)
        cost = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=y, pos_weight=pos_weight)
        cost = tf.reduce_mean(cost * (1 - beta))
        return tf.where(tf.equal(count_pos, 0.0), 0.0, cost, name=name)
    
    def hedLoss(self, b1, b2, b3, b4, b5, predEdgeMap, target):
        costs=[]
        for idx, b in enumerate([b1, b2, b3, b4, b5, predEdgeMap]):
            output = tf.nn.sigmoid(b, name='output{}'.format(idx + 1))
            xentropy = self.class_balanced_sigmoid_cross_entropy(
                b, target,
                name='xentropy{}'.format(idx + 1))
            costs.append(xentropy)
        
        # some magic threshold
        pred = tf.cast(tf.greater(output, 0.5), tf.int32, name='prediction')
        tr_errors = tf.cast(tf.not_equal(pred, target), tf.float32)
        trainErrors = tf.reduce_mean(tr_errors, name='train_error')
        
        wd_w = tf.train.exponential_decay(2e-4, tf.train.get_or_create_global_step(),  # this is borrowed from tensorpack, it is not for learning rate
                                              80000, 0.7, True)                        # it is for the coefficient (lambda_i) of weight regularization
        
        G = tf.get_default_graph()
        params = G.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for param in params:
            if param.name.find('wt'):
                l2_norm_cost = tf.multiply(wd_w, tf.nn.l2_loss(param), name='wd_l2_norm') 
                costs.append(l2_norm_cost)
        
        totalCost = tf.add_n(costs, name='totalCost')
        return totalCost, trainErrors
    
    def trainModel(self, hedModel = None, preTrainedHedModelPath="../hedPreTrainedModel/HED_reproduced.npy",
                    pretrainedCellDetModelPath = "../checkpoint/model.ckpt-16", trainDataPath = "../data/trainData", valDataPath = "../data/valData", 
                    batchSize = 8, trainDataMean = 188.9): 
        
        tf.reset_default_graph()
        sess = tf.Session()
        
        x = tf.placeholder(tf.float32, shape=[None, 304*304*3], name="inputX")
        x_image = tf.reshape(x, [-1, 304, 304, 3])
        tf.summary.image('input', x_image, 3)
        y = tf.placeholder(tf.int32, shape=[None, 304,304,1], name="edgemap")
        
        b1_pred, b2_pred, b3_pred, b4_pred, b5_pred, branchComb_pred  = self.createHed(x_image)
        trainLoss, pixelErrors = self.hedLoss(b1_pred, b2_pred, b3_pred, b4_pred, b5_pred, branchComb_pred, y)
        
        with tf.name_scope("train"):
            
            global_step = tf.train.get_global_step()
            starter_learning_rate = 0.00003
            learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                      10000, 0.1, staircase=True)    
            opt = tf.train.AdamOptimizer(learning_rate=starter_learning_rate, epsilon=1e-3)
            train_op = opt.minimize(trainLoss, global_step=global_step)
            
            # Build an initialization operation to run below.
            init = tf.global_variables_initializer()
            variables = tf.global_variables()
            
            # Build a saver
            saver = tf.train.Saver(tf.global_variables(), max_to_keep = 10)
            summ = tf.summary.merge_all()
                    
            sess.run(init)
            if os.path.isfile(pretrainedCellDetModelPath):
                saver.restore(sess, pretrainedCellDetModelPath)
                print('train from existing cell detection model')
            else:
                G = tf.get_default_graph()
                params = G.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                getPretrainedModel(myModelParams=params, sess=sess) # set the parameters using the pretrained VGG16 model
                print('train from pretrained HED model')
            
            trDataPath = trainDataPath
            xPath = os.path.join(trDataPath, 'trainPatches')
            yPath = os.path.join(trDataPath, 'trainMasks')
            xPaths = glob.glob(xPath + '/*.jpg')
            xPaths = sorted(xPaths)
            yPaths = glob.glob(yPath + '/*.png')
            yPaths = sorted(yPaths)
            noSamples = len(xPaths)
            
            xPathVal = os.path.join(valDataPath, 'trainPatches')
            yPathVal = os.path.join(valDataPath, 'trainMasks')
            xPathsVals = glob.glob(xPathVal + '/*.jpg')
            xPathsVals = sorted(xPathsVals)
            yPathsVals = glob.glob(yPathVal + '/*.png')
            yPathsVals = sorted(yPathsVals)
            noVal = len(xPathsVals)
            
            global_cnt = 0
            for i in range(100000):
                id = 0
                epochLoss = 0
                while id < noSamples-batchSize:
                    batchX = []
                    batchY = []
                    for j in range(batchSize):
                        randomIndx = np.random.randint(low=0, high=noSamples, size=1)
                        im = cv2.imread(xPaths[randomIndx[0]], 0)
                        im = np.asarray(im, dtype = np.float32)
                        im = im - trainDataMean
                        im = np.expand_dims(im, axis=2)
                        im = np.concatenate((im, im, im), axis=2)
                        batchX.append(im)
                        
                        maskY = cv2.imread(yPaths[randomIndx[0]], 0)
                        maskY = np.asarray(maskY, dtype=np.int32)
                        maskY = np.expand_dims(maskY, axis=2)
                        batchY.append(maskY)
                    id = id + batchSize
                    
                    _,train_loss, summ_ = sess.run([train_op, trainLoss, summ], feed_dict={x_image: batchX, y: batchY})
                    epochLoss += train_loss
                    
                    global_cnt += 1
                    
                    
                print("epoch: {}".format(i))
                print("loss: {}".format(epochLoss/(noSamples/batchSize)))
                    
                
                if i % 5 ==0:
                    saver.save(sess, '../checkpoint/model.ckpt', i)
                    
                    batchX = []
                    batchY = []
                    for j in range(noVal):
                        
                        im = cv2.imread(xPathsVals[j], 0)
                        im = np.asarray(im, dtype = np.float32)
                        im = im - trainDataMean
                        im = np.expand_dims(im, axis=2)
                        im = np.concatenate((im, im, im), axis=2)
                        batchX.append(im)
                        
                        maskY = cv2.imread(yPaths[j], 0)
                        maskY = np.asarray(maskY, dtype=np.int32)
                        maskY = np.expand_dims(maskY, axis=2)
                        batchY.append(maskY)
                    trainErrors = sess.run(pixelErrors, feed_dict={x_image: batchX, y:batchY})
                    print("train errors: {}".format(trainErrors))
    
    def modelPred(self, modelFilePath = '../checkpoint/model.ckpt-165', testImgFolder = '../data/testData/', 
                  testOneFile = False, testImgPath = None, resultFolder = '../data/result/', trainDataMean = 188.9):
        # if want to test just on image, set testOneFile = True, and provide image path and name by testImgPath = path/to/img.jpg
         
        tf.reset_default_graph()
        sess = tf.Session()        
        
        x = tf.placeholder(tf.float32, shape=[None, 304*304*3], name="inputX")
        x_image = tf.reshape(x, [-1, 304, 304, 3])
    
        _, _, _, _,_, branchComb_pred  = self.createHed(x_image)
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess, modelFilePath)
        
        if testOneFile == False: 
            xPath = os.path.join(testImgFolder)
            xPaths = glob.glob(xPath + '/*.jpg')
            xPaths = sorted(xPaths)
        else:
            xPaths = [testImgPath]
        
        testImgs = []
        testImgNames = []
        for imgPath in xPaths:
            filePath, imgName = os.path.split(imgPath)
            img = cv2.imread(imgPath)
            
            img = np.asarray(img, dtype=np.float32)
            img[:,:,0] = (img[:,:,0] - trainDataMean)
            img[:,:,1] = (img[:,:,1] - trainDataMean)
            img[:,:,2] = (img[:,:,2] - trainDataMean)
            
            
            testImgs.append(img)
            testImgNames.append([filePath, imgName])
            
        cellDet = sess.run(branchComb_pred, feed_dict={x_image:[testImgs[0]]})
        
        for i in range(len(cellDet)):
            oneDet = np.squeeze(cellDet[i])
            rawMask = [oneDet>0.8]
            rawMask = np.where(rawMask, 1, 0)
            rawMask = np.squeeze(rawMask)
            rawMask = rawMask.astype(int)
            distance = ndimage.distance_transform_edt(rawMask)
            local_maxi = peak_local_max(distance, min_distance=5)
            local_maxi_oneImg = np.empty( shape=(0,2) )
            if len(local_maxi)>0:
                local_maxi_oneImg = local_maxi
                ms = MeanShift(bandwidth=10.0, bin_seeding=True)
                ms.fit(local_maxi_oneImg)
                
                cluster_centers = ms.cluster_centers_
                
                seedsOneImg = cluster_centers
                
                filePath, imgName = testImgNames[i]
                im = Image.open(os.path.join(filePath, imgName))
                
                imWdt, imHt = im.size
                if len(np.shape(im)) == 2:
                    im = np.array(im, dtype=np.float32)
                    #im_cv = im[:,:, ::-1].copy()
                    im_cv = np.zeros((imWdt, imHt, 3))
                    im_cv[:,:,0] = im.copy()
                    im_cv[:,:,1] = im.copy()
                    im_cv[:,:,2] = im.copy()
                elif len(np.shape(im)) == 3:
                    im = np.array(im, dtype=np.float32)
                    #im_cv = im[:,:, ::-1].copy()
                    im_cv = np.zeros((imWdt, imHt, 3))
                    im_cv[:,:,0] = im[:,:,0].copy()
                    im_cv[:,:,1] = im[:,:,1].copy()
                    im_cv[:,:,2] = im[:,:,2].copy()
                            
                temp = np.zeros((seedsOneImg.shape[0], seedsOneImg.shape[1]))        
                temp[:,1]=seedsOneImg[:,0]
                temp[:,0]=seedsOneImg[:,1] 
                for ss in temp:
                    cv2.circle(im_cv,tuple(np.int32(ss)), 2, (0, 255, 0), thickness=2)    
                cv2.imwrite(resultFolder + imgName[:-3] + 'jpg', im_cv)    
                
        