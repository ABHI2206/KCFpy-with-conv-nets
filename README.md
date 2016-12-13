# KCF tracker in Python

Python implementation of
> [High-Speed Tracking with Kernelized Correlation Filters](http://www.robots.ox.ac.uk/~joao/publications/henriques_tpami2015.pdf)<br>
> J. F. Henriques, R. Caseiro, P. Martins, J. Batista<br>
> TPAMI 2015

It is fix version for python 3 of [KCFpy](https://github.com/uoip/KCFpy).


It is translated from [KCFcpp](https://github.com/joaofaro/KCFcpp) (Authors: Joao Faro, Christian Bailer, Joao F. Henriques), a C++ implementation of Kernelized Correlation Filters. Find more references and code of KCF at http://www.robots.ox.ac.uk/~joao/circulant/

## Requirements For KCF Tracker
- Python 3
- NumPy
- Numba (needed if you want to use the hog feature)
- OpenCV (ensure that you can `import cv2` in python)

## Requirements For KCF Tracker with Vggnet 16
- Python 3
- NumPy
- Numba (needed if you want to use the hog feature)
- OpenCV (ensure that you can `import cv2` in python)
- Theano
- Pickle
- Lasagne


## Use

### For KCF Tracker
Download the sources and execute
```shell
git clone https://github.com/Sshanu/KCFpy.git
cd KCFpy
python run_updated.py
```
It will open the default camera of your computer, you can also open a video
```shell
python run_updated.py -inv test.avi  
```
### For KCF Tracker with Vggnet 16
Download pretrained weights from: https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl
and save in new_vgg16 folder.
Then execute
```shell
python run_cnn.py -inv test.avi -opt Output_Folder -mo cnn
```
or For Webcam
```shell
python run_cnn.py -opt Output_Folder -mo cnn
```

For using GPU to compute conv layer :
change line 14 of vgg16.py in new_vgg16 folder
```
from lasagne.layers import Conv2DLayer as ConvLayer
```
to
```
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
```
