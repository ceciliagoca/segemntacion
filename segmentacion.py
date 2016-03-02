# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.


__author__ = "cecilia"
__date__ = "$Feb 23, 2016 11:05:50 AM$"
__update__ = "$Mar 02, 2016 11:05:50 AM$"
import numpy as np
import cv2 as cv


b_showing = 0   ;
k =3;
max_iter =5 ;
eps =.03;
sh_scale =.30;


def imgResize(img,p):
    cols, rows = img.shape[:2];
    ds_cols = int(p*cols);
    ds_rows = int(p*rows);
    ds_img = cv.resize(img, (ds_rows, ds_cols), interpolation = cv.INTER_CUBIC)
    return ds_img


def imgShow(img, p, str):
    ds_img=imgResize(img,p)
    cv.imshow(str,ds_img);
    cv.waitKey(0);
    return 0


def getNDVI(img):

    #get channels
    #d_x,d_blue,d_nir = cv.split(img)
    #b: nothin g: blue r: nir

    d_blue = img[:,:,1]
    d_nir = img[:,:,2]
    print(d_blue.dtype)
    rows, cols = img.shape[:2];
    ndvi = np.zeros((rows,cols,1),np.float32)
    aux_nir = np.zeros((rows,cols,1),np.float32)
    aux_blue = np.zeros((rows,cols,1),np.float32)
    ##falta calibrar el azul
    aux_nir = d_nir/255.0;
    aux_blue = d_blue/255.0;
    ndvi = (aux_nir - aux_blue)/(aux_nir + aux_blue);
    return aux_nir


def segmetarByKMeans( img,  k,  max_iter,  eps):
    print (img.shape)
    rows, cols = img.shape[:2];
    p = img.reshape(( cols*rows,1));
    p = np.float32(p)

    #attempts  Flag to specify the number of times the algorithm is executed using different initial labellings.
    attempts =10;
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, max_iter, eps)
    ret,label,center=cv.kmeans(p,k,None,criteria,attempts,cv.KMEANS_RANDOM_CENTERS)

    #a = np.where(label == 0)[0];  #indoces donde se cumple la restricción
    #print ('a: ' , a);ee ee
    b= 255 / k;
    for x in range(0,k):
        label[label == x] = int(x*b)



    label = label.reshape(img.shape)
    img_seg = np.uint8(label)

    cv.imshow('dddd', img);
    cv.imshow('dddad', img_seg);
    cv.waitKey(0);

    return img_seg


if __name__ == "__main__":
    print ("Segmentacion de plantas con k means")
    print ("version de opencv: " + cv.__version__)

#img = cv.imread('datos/lechuga_ndvi/1.JPG',1)
img = cv.imread('datos/ndvis/1.tif',cv.IMREAD_ANYDEPTH)
img = imgResize(img,sh_scale)
#cv.imshow('ndvi',img);
#cv.waitKey(0);
sh_scale =1;


# if b_showing:
#     imgShow(img, sh_scale, "original")
#
# ndvi=getNDVI(img)
#
# if b_showing:
#     imgShow(ndvi, sh_scale, "ndvi")
#
# segRes = segmetarByKMeans(ndvi, k,  max_iter,  eps);
segRes = segmetarByKMeans(img, k,  max_iter,  eps);



if b_showing:
    imgShow(segRes, sh_scale, "segementacion_1")

#
#
# if b_showing:
#     imgShow(img, sh_scale, "original")
#
# ndvi=getNDVI(img)
#
# if b_showing:
#     imgShow(ndvi, sh_scale, "ndvi")
#
# segRes = segmetarByKMeans(ndvi, k,  max_iter,  eps);
#
# if b_showing:
#     imgShow(segRes, sh_scale, "segementacion_1")
#






