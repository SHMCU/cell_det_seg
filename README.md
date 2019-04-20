# cell_det_seg

This project is for nucleus/cell detection and segmentation in pathology images.
This project includes three networks: 1) HED network implemented using Tensorflow; 2) nuclues/cell size prediction network based on VGG16 implemented using Keras and Tensorflow;  3) Nucleus/cell segmentation based on convolutional variational autoencoder trained in a multi-task learning setting. The system runs faster than MaskRCNN network and produces similar performance.
More details are on the way.

models for the three networks used in this project can be found here:
https://www.dropbox.com/sh/43cg7s3at9z5pqq/AADKPhDUDTwVGWj56oayh8tua?dl=0

Requirements:
Tensorflow = 1.11.0
Keras = 2.2.4
Python3.6

How to run:
1) Download models from the dropbox link. Put the folder "model" in the root folder.
2) run by : 
	python Demo.py
3) The detection results are saved to the folder: /data/cell_det_data/res/3/
The segmentation results are saved to /data/cell_seg_data/res/3 

To test on your data:
1) Make sure the testing images are H&E stained pathology images
2) The size of the cells are roughly between 20~90 pixels for an image of size 1000x1000 or so.

