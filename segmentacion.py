# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

__author__ = "cecilia"
__date__ = "$Feb 23, 2016 11:05:50 AM$"
__update__ = "$Mar 02, 2016 11:05:50 AM$"
import numpy as np
import cv2 as cv

b_showing = 1   ;
k =3;
max_iter = 15 ;
eps =.001        ;
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


def segmetarByKMeans(img, k, max_iter, eps):
    print (img.shape)
    rows, cols = img.shape[:2];
    p = img.reshape(( cols*rows,1));
    p = np.float32(p)

    #attempts  Flag to specify the number of times the algorithm is executed using different initial labellings.
    attempts =10;
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, max_iter, eps)
    ret,label,center=cv.kmeans(p,k,None,criteria,attempts,cv.KMEANS_RANDOM_CENTERS)


    b = 255 / (k-1);
    #
    # label[label == 0] = 0*b
    # label[label == 1] = 1*b
    # label[label == 2] = 2*b

    #for x in range(0,k):
    #    label[label == x] = x*b

    a = sorted(range(len(center)), key=lambda k: center[k]) #indices oordenados de menor a mayor
    print(a)

    aux_blue = label.copy()
    for x in range(0,k):
        aux_blue[label == a[x]] = (x*b)
        print (x)


    aux_blue = aux_blue.reshape(img.shape)
    img_seg = np.uint8(aux_blue)

    return img_seg

    #def postSegmentacion(img)
    #find contours
def findCountorns(img):
    rows, cols = img.shape[:2];
    print(img.shape)
    thresh1 = np.zeros((rows,cols,1),np.uint8)
    #img2 =  cv.cvtColor(img,cv.COLOR_GRAY2RGB)
    ret,thresh1 = cv.threshold(img,1 ,255,cv.THRESH_BINARY)
    np.uint8(thresh1)
    image, contours, hierarchy = cv.findContours(thresh1,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    img_ = cv.drawContours(img, contours, -1, (0,255,0), 3)

    cv.imshow( 'countorns', img_ )
    return img_


def segmentar(img,seg):
    iseg = seg.copy();

    rows, cols = img.shape[:2];
    newimg = np.zeros((rows,cols,1),np.uint8)

    ret,thresh1 = cv.threshold(iseg,1 ,255,cv.THRESH_BINARY)
    #img2_fg = cv.bitwise_and(img,img,thresh1)
    #img1_bg = cv.bitwise _and(img,thresh1)
    img1 = cv.bitwise_and(img, img, mask=thresh1)
    cv.imshow('cany1', img1)

    #findCountorns(img1);
    #kernel = np.ones((9,9),np.uint8)
    #opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel,2)
    #erosion = cv.erode(thresh1,kernel,iterations =1)

    #edges = cv.Canny(thresh1,100,255)

    #kernel = np.ones((9,9),np.uint8)
    #gradient = cv.morphologyEx(erosion, cv.MORPH_GRADIENT, kernel)

    #cv.imshow('cany1', thresh1)
    #cv.imshow('cany2', opening)
    #cv.imshow('cany3', gradient)

    #suma = gradient + erosion
    #cv.imshow('cany4', suma)
    cv.waitKey(0)


   # kernel = cv.getStructuringElement(cv.MORPH_CROSS,(10,10 ))



    opening = cv.morphologyEx(iseg, cv.MORPH_DILATE, kernel)
    return iseg


def watershed(img):
    rows, cols = img.shape[:2];
    img2 = np.zeros((rows,cols,3),np.uint8)
    img2 =  cv.cvtColor(img,cv.COLOR_GRAY2RGB)

    ret, thresh = cv.threshold(img,1,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    cv.imshow('treshh', thresh)
    cv.waitKey(0)


    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)

    # sure background area
    sure_bg = cv.dilate(opening,kernel,iterations=3)

    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
    ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg,sure_fg)

    cv.imshow('treshh2', unknown)
    cv.waitKey(0)

    ret, markers = cv.connectedComponents(sure_fg)
    markers = markers+1
    markers[unknown ==255] = 0
    markers = cv.watershed(img2,markers)
    img2[markers == -1] = [255,0,0]


    cv.imshow('treshh3', img2)
    cv.waitKey(0)


    return thresh


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


segmentar(img,segRes)
##watershed(segRes)
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






