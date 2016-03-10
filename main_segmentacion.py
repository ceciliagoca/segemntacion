__author__ = "cecilia"
__date__ = "$Feb 23, 2016 11:05:50 AM$"
__update__ = "$Mar 03, 2016 11:05:50 AM$"

import numpy as np
import cv2 as cv
import utilsImg as u
import segmentacion as s

b_showing = 1;
sh_scale =.30;

def getNDVI(img):

    d_blue = img[:,:,1]
    d_nir = img[:,:,2]
    ##falta calibrar el azul
    aux_nir = d_nir/255.0;
    aux_blue = d_blue/255.0;
    ndvi = (aux_nir - aux_blue)/(aux_nir + aux_blue);
    #retorno el nir porque funciona mejor que ndvi
    return ndvi


if __name__ == "__main__":
      print ("Segmentacion de plantas con k means")
      print ("version de opencv: " + cv.__version__)

#img = cv.imread('datos/lechuga_ndvi/1.JPG',1)
#ndvi = getNDVI(img)

##leyendo NDVI
img = cv.imread('datos/ndvis/1.tif',cv.IMREAD_ANYDEPTH)
img = u.imgResize(img,sh_scale)

sh_scale =1;

if b_showing:
    u.imgShow(img, sh_scale, "ndvi")

#separar tierra de plantas, primera segmentación
k = 4;
max_iter = 3;
eps =.001;
segRes = s.segmetarByKMeans(img, k,  max_iter,  eps);
ret, mask = cv.threshold(segRes,1,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
masked_data = cv.bitwise_and(img, img, mask=mask)
aux = s.segmentar(masked_data)
cv.imshow('sss', aux);
cv.waitKey(0);