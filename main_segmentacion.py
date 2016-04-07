__author__ = "cecilia"
__date__ = "$Feb 23, 2016 11:05:50 AM$"
__update__ = "$Mar 03, 2016 11:05:50 AM$"

import numpy as np
import cv2 as cv
import utilsImg as u
import segmentacion as s

b_showing = 1;
sh_scale =.25;

def getNDVI(img):

    d_blue = img[:,:,1]
    d_nir = img[:,:,2]
    aux_nir = d_nir/255.0;
    aux_blue = d_blue/255.0;
    ndvi = (aux_nir - aux_blue)/(aux_nir + aux_blue);



    #retorno el nir porque funciona mejor que ndvi
    return ndvi


def segmentacion_01(img_ndvi): #unicamente segmetacion

    veg_mask = s.segOtsu(img_ndvi)
    img_veg = cv.bitwise_and(img_ndvi,img_ndvi,mask = veg_mask)
    img_veg = img_veg * 255;

    img_veg = np.uint8(img_veg)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(img_veg)
    cv.imshow('vegetaciónCL', cl1)



    opening = cv.morphologyEx(cl1, cv.MORPH_OPEN,  cv.getStructuringElement(cv.MORPH_ELLIPSE,(7,7)))
    #opening = cv.GaussianBlur(opening,(7,7),0)
    #opening=cv.blur(opening,(5,5))
    opening=cv.medianBlur(opening,7)

    opening [opening>135] =0
    laplacian = cv.Laplacian(opening,cv.CV_64F,ksize=1, scale=.05 , borderType=cv.BORDER_DEFAULT) #bordes de uniones los voy a enogrodar y a restarlos a la imagen original
    laplacian = cv.dilate(laplacian,cv.getStructuringElement(cv.MORPH_ELLIPSE,(7,7)));

    newimg = opening #np.zeros((907,1209),np.uint8);
    newimg [laplacian > .9 ] = 0

    cv.imshow('laplacian', laplacian)
    cv.imshow('segmentacion_03', newimg)

    #erdoe =  cv.erode(newimg,cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5)))
    #cv.imshow('segmentacion_03', erdoe)

    opening = cv.morphologyEx(newimg, cv.MORPH_CLOSE,  cv.getStructuringElement(cv.MORPH_ELLIPSE,(7,7)))
    opening = cv.morphologyEx(opening, cv.MORPH_CLOSE,  cv.getStructuringElement(cv.MORPH_ELLIPSE,(7,7)))
    cv.imshow('segmentacion_4', opening)


    #opening = cv.threshold(opening,.8,1,cv.THRESH_BINARY)
     # Marker labelling
    ret, markers = cv.connectedComponents(opening)


    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[veg_mask==0] = 0

    vegMask = img_ndvi*255
    vegMask = np.uint8(vegMask)

    vegMask = cv.cvtColor(opening,cv.COLOR_GRAY2BGR );

    markers = cv.watershed(vegMask,markers)
    vegMask[markers == -1] = [255,0,0]

    cv.imshow('segmentacion', vegMask)
    cv.waitKey(0)
    return 0


def segmentacion_02(img_ndvi): #wahteshed

    veg_mask = s.segOtsu(img_ndvi)

    kernel = np.ones((5,5),np.uint8)
    veg_mask = cv.morphologyEx(veg_mask,cv.MORPH_OPEN,kernel, iterations = 3)


    img_veg = cv.bitwise_and(img_ndvi,img_ndvi,mask = veg_mask)
    img_veg = img_veg * 255;
    vegMask = np.uint8(img_veg)
    vegMask = cv.cvtColor(vegMask,cv.COLOR_GRAY2BGR );

    #
    # kernel = np.ones((5,5),np.uint8)
    # opening = cv.morphologyEx(veg_mask,cv.MORPH_OPEN,kernel, iterations = 3)
    #
    #
    # # sure background area
    # sure_bg = cv.dilate(opening,kernel,iterations=3)
    #
    # # Finding sure foreground area
    # dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
    # ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    #
    # # Finding unknown region
    # sure_fg = np.uint8(sure_fg)
    # unknown = cv.subtract(sure_bg,sure_fg)


    # Marker labelling
    ret, markers = cv.connectedComponents(veg_mask)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[veg_mask==0] = 0



    markers = cv.watershed(vegMask,markers)
    vegMask[markers == -1] = [255,0,0]


    cv.imshow('maskara',veg_mask);
    # cv.imshow('openin',opening);
    # cv.imshow('sure_bg',sure_bg);
    # cv.imshow('dist_trans',dist_transform);
    # cv.imshow('sure_fg',sure_fg);
    # cv.imshow('unknow', unknown);
    # cv.imshow('markers', markers);
    cv.imshow('ffff', vegMask);
    cv.waitKey();
    return 0

def segmetacion_03(img_nir, img_ndvi): #con kmeans

    #d_blue = img_nir[:,:,1]
    d_nir = img_nir[:,:,2]
    veg_mask = s.segOtsu(img_ndvi)

    #mascara de nir y de ndvi
    veg_zones_ndvi = cv.bitwise_and(img_ndvi,img_ndvi,mask = veg_mask)
    veg_zones_nir = cv.bitwise_and(d_nir,d_nir,mask = veg_mask)
    veg_zones_nir_f32 = np.float32(veg_zones_nir);
    veg_zones_nir_f32 = veg_zones_nir_f32/255;
    k = 4;
    max_iter = 5;
    eps =.01;

    rows, cols = veg_zones_ndvi.shape[:2];
    p = np.zeros(( cols*rows,2));
    p[:,0] = veg_zones_ndvi.reshape(( cols*rows,1))[:,0];
    veg_zones_nir_f32 = veg_zones_nir_f32/255
    p[:,1] = veg_zones_nir_f32.reshape(( cols*rows,1))[:,0];
    p = np.float32(p)
    x = s.segmetarByKMeans(img_ndvi,p,k,max_iter,eps)
    cv.imshow('kmeans', x);
    cv.waitKey();

    return 0

def segmentacion_04(img):

    original = img_nir[:,:,2]


    veg_mask = s.segOtsu(original,11)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    seem = cv.morphologyEx(veg_mask, cv.MORPH_DILATE, kernel, iterations=1)

    seem = cv.morphologyEx(veg_mask, cv.MORPH_ERODE, kernel, iterations=1)

    #img_e = cv.equalizeHist(original)

    # create a CLAHE object (Arguments are optional).
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_e = clahe.apply(original)

    img_veg = cv.bitwise_and(img_e, img_e, mask=seem)


    cv.imshow('original',original)
    cv.imshow('veg_mask', veg_mask)

    ##crear semilas
    dist = cv.distanceTransform(veg_mask, cv.DIST_L2, cv.DIST_MASK_PRECISE)  # distanceTransform(bw, dist, CV_DIST_L2, 3);
    cv.normalize(dist, dist, 0, 1., cv.NORM_MINMAX);
    dist = dist * 255
    dist = np.uint8(dist)
    cv.imshow('seem0', dist)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))
    seem = cv.morphologyEx(dist, cv.MORPH_ERODE, kernel, iterations=4)
    #kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    cv.imshow('seem1', seem)

    cv.waitKey(0)
    #cv.destroyAllWindows()

    #apilcar watershe
    s.watershed2(seem,img_veg)
    cv.waitKey();


    return 0


def segmentacion_05(img,res): # img = imagen en escala de grises

    # lectura de imagen
    if img.dtype != np.uint8 :
        img = img * 255
        original = np.uint8(img)
    else:
        original = img


    #

    k_otsu = 11 if res==1 else  5;
    k_omorf = 6  if res==1 else 3;


    # Segmentar tierra y vegetación
    veg_mask = s.segOtsu(original, k_otsu)
    cv.imshow('veg_mask', veg_mask)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k_omorf, k_omorf))
    veg_mask = cv.morphologyEx(veg_mask, cv.MORPH_CLOSE, kernel, iterations=1)
    cv.imshow('veg_mask2', veg_mask)
    # create a CLAHE object (Arguments are optional).
    #clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #img_e = clahe.apply(original)
    img_e = cv.equalizeHist(original)
    img_veg = cv.bitwise_and(img_e, img_e, mask=veg_mask)
    cv.imshow('img_veg', img_veg)
    cv.waitKey()

    ##crear semilas
    dist = cv.distanceTransform(veg_mask , cv.DIST_L2, 3)  # distanceTransform(bw, dist, CV_DIST_L2, 3);
    cv.normalize(dist, dist, 0, 1., cv.NORM_MINMAX);
    dist = dist * 255
    dist = np.uint8(dist)
    cv.imshow('seem0', dist)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k_omorf*2, k_omorf*2))
    seem = cv.morphologyEx(dist, cv.MORPH_ERODE, kernel, iterations=4)
    seem = cv.morphologyEx(seem, cv.MORPH_CLOSE, kernel, iterations=4)
    cv.imshow('seem1', seem)
    cv.waitKey(0)

    # apilcar watershe
    s.watershed2(seem, img_veg)
    cv.waitKey();

    return 0;
if __name__ == "__main__":
      print ("Segmentacion de lechugas")
      print ("version de opencv: " + cv.__version__)


img_nir = cv.imread('datos/lechuga_ndvi/8.JPG',1)
img_nir = u.imgResize(img_nir,sh_scale)
d_nir = img_nir[:,:,2]

img_ndvi = cv.imread('datos/ndvis/1.tif',cv.IMREAD_ANYDEPTH)
img_ndvi = u.imgResize(img_ndvi,sh_scale)


#cv.imshow('ndvi',img_ndvi)

#conjunto de caracteristicas (ndvi, nir)

f_nir = np.float32(d_nir);
f_nir = f_nir/255;

rows, cols = f_nir.shape[:2];
p = np.zeros(( cols*rows,2));
p[:,0] = f_nir.reshape(( cols*rows,1))[:,0];
p[:,1] = img_ndvi.reshape(( cols*rows,1))[:,0];
p = np.float32(p)


#segmentacion_05(img_nir[:, :, 2])
segmentacion_05(img_nir[:,:,2], res=0) # 1 <<- res : resolucion alta ; 0 resolucion de vuelo
#segmentacion_01(img_ndvi)
#segmentacion_02(img_ndvi)
#segmetacion_03(img_nir,img_ndvi)

#s.watershed(img_nir[:,:,2])

#s.clouster_SCL(p)
#a  = s.MiniBachkMeans(img_nir,p);
#a= s.AffinityPropagation(img_nir,p);

#=cv.imshow("segmentacion_result", a)

#cv.waitKey(0);