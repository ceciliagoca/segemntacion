__author__ = "cecilia"
__date__ = "$Feb 23, 2016 11:05:50 AM$"
__update__ = "$Mar 03, 2016 11:05:50 AM$"
import numpy as np
import cv2 as cv


b_showing = 1   ;
k =2
max_iter = 15;
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

    d_blue = img[:,:,1]
    d_nir = img[:,:,2]
    print(d_blue.dtype)
    ##falta calibrar el azul
    aux_nir = d_nir/255.0;
    aux_blue = d_blue/255.0;
    ndvi = (aux_nir - aux_blue)/(aux_nir + aux_blue);
    #retorno el nir porque funciona mejor que ndvi
    return aux_nir

def segmetarByKMeans(img, k, max_iter, eps):

    rows, cols = img.shape[:2];
    p = img.reshape(( cols*rows,1));
    p = np.float32(p)

    #attempts  Flag to specify the number of times the algorithm is executed using different initial labellings.
    attempts =10;
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, max_iter, eps)
    ret,label,center=cv.kmeans(p,k,None,criteria,attempts,cv.KMEANS_RANDOM_CENTERS)

    a = sorted(range(len(center)), key=lambda k: center[k]) #indices oordenados de menor a mayor

    b = 255 / (k-1);
    aux = label.copy()

    for x in range(0,k):
        aux[label == a[x]] = (x*b)
        print (x)

    aux = aux.reshape(img.shape)
    img_seg = np.uint8(aux)

    return img_seg

def findCountorns(img, img_bin): #recibe imagen en escala de grises y binaris

    im2, contours, hierarchy = cv.findContours(img_bin,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img, contours, -1, (0,255,0), 3)

    return img

def segmentar(img):

    rows, cols = img.shape[:2];
    ret,thresh1 = cv.threshold(img,1 ,255,cv.THRESH_BINARY)

    return thresh1

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
