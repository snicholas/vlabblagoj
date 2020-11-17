"""
Title: Test for using tensorflow on VLAB
Authors: Blagoj Delipetrev, Mattia Santoro, Nicholas Spadaro
Date created: 2020/11/09
Last modified: 2020/11/09
Description: Templates for creation and execution through the VLAB on DestinationEarth VirtualCloud of random forest based digital twins.
Version: 0.1
"""
import os, sys
from osgeo import gdal
from osgeo import gdalconst
import numpy as np
import fnmatch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm

from tensorflow.keras import layers
from tensorflow import keras

from keras.models import Model, load_model
from keras.layers import Input
from keras import backend as K
from keras.layers import merge, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Lambda
from keras.optimizers import Adam
from keras.layers.merge import concatenate

from model import unet

def ReadImage(fname, readflag=1):
    src = gdal.Open(fname)
    projection = src.GetProjection()
    geotransform = src.GetGeoTransform()
    datatype = src.GetRasterBand(1).DataType
    datatype = gdal.GetDataTypeName(datatype)
    ulx, xres, xskew, uly, yskew, yres  = src.GetGeoTransform()
    lrx = ulx + (src.RasterXSize * xres)
    lry = uly + (src.RasterYSize * yres)
    cols = src.RasterXSize
    rows = src.RasterYSize
    Image = 0
    if readflag:
        Image = src.GetRasterBand(1).ReadAsArray()
        print('Image shape: %d %d' % (Image.shape))
    print('Spatial resolution: %f %f' % (xres,yres))
    return Image, projection, geotransform, (ulx,uly), (lrx,lry), src

def Setcmap(D, inv=1):
    cmap = plt.cm.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmaplist[0] = (.0, .0, .0, 1.0)
    if inv:
        cmap = LinearSegmentedColormap.from_list('Custom_cmap', cmaplist, cmap.N)
    else:
        cmap = LinearSegmentedColormap.from_list('Custom_cmap', cmaplist[::-1], cmap.N)
    plt.register_cmap(name='Custom_cmap', cmap=cmap)
    bounds = np.linspace(0, D, D+1)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm
    
def blockshaped(arr, nrows, ncols):
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

def unblockshaped(arr, h, w):
    n, nrows, ncols = arr.shape
    return (arr.reshape(h//nrows, -1, nrows, ncols)
               .swapaxes(1,2)
               .reshape(h, w))

# mettere parametro nome immagine?
folder = 'scene.SAFE'
infolder = os.path.join(folder, 'GRANULE')
infolder = os.path.join(infolder, os.listdir(infolder)[0], 'IMG_DATA')
infile = os.path.join(infolder, fnmatch.filter(os.listdir(infolder), '*B02.jp2')[0])
print('Reading '+infile)
Blue, PROJ, GEOTR, p1, p2, SRC = ReadImage(infile)
rows, cols = SRC.RasterYSize, SRC.RasterXSize
infile = os.path.join(infolder, fnmatch.filter(os.listdir(infolder), '*B03.jp2')[0])
print('Reading '+infile)
Green, _, _, _, _, _ = ReadImage(infile)
infile = os.path.join(infolder, fnmatch.filter(os.listdir(infolder), '*B04.jp2')[0])
print('Reading '+infile)
Red, _, _, _, _, _ = ReadImage(infile)
infile = os.path.join(infolder, fnmatch.filter(os.listdir(infolder), '*B08.jp2')[0])
print('Reading '+infile)
NIR, _, _, _, _, _ = ReadImage(infile)

window = 244
A = blockshaped(Blue, window, window)
Stack = np.expand_dims(A, axis=0)
A = blockshaped(Green, window, window)
Stack = np.concatenate((Stack, np.expand_dims(A, axis=0)), axis=0)
A = blockshaped(Red, window, window)
Stack = np.concatenate((Stack, np.expand_dims(A, axis=0)), axis=0)
A = blockshaped(NIR, window, window)
Stack = np.concatenate((Stack, np.expand_dims(A, axis=0)), axis=0)
del A, Blue, Green, Red, NIR
Stack.shape

X = np.float32(Stack)
X[X>10000] = 10000
X = X / 10000.
X = np.rollaxis(X, 0, 4)
X.shape

X=X[:,:128,:128,:]

std_params = np.load('std_params.npy')
mu = std_params[0,:]
std = std_params[1,:]

for q in range(X.shape[3]):
    tmp = np.copy(X[:, :, :, q])
    tmp = tmp.flatten()
    print(mu[q], std[q])
    tmp[tmp>0] = (tmp[tmp>0] - mu[q]) / std[q]
    X[:, :, :, q] = tmp.reshape(X.shape[0], X.shape[1], X.shape[2])
    del tmp

img_size=(128, 128, 4)
num_classes = 9


keras.backend.clear_session()

# Build model
model = unet(img_size, num_classes)
model.summary()

input_shape = (128, 128, 4)
model = unet(input_shape,9)
model.summary()
batch_size = 25
fnmodel = 'model.h5'
model.load_weights(filepath = fnmodel)

Resp = model.predict(X, batch_size=3*batch_size).argmax(axis=-1)
Class = unblockshaped(Resp, 5760, 5760)

cmap, norm = Setcmap(9)
im = plt.imshow(Class, cmap=cmap, norm=norm, vmin=0, vmax=9)
plt.title('Classification')
plt.set_cmap(cmap)
cb = plt.colorbar(im, fraction=0.046, pad=0.04, ticks=range(9))
cb.set_ticks(np.arange(9) + .5)
cb.set_ticklabels(np.arange(9))
plt.gcf().set_size_inches(15, 14)
plt.savefig("data/outputs/result.png")
