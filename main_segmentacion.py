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

img_nir = cv.imread('datos/lechuga_ndvi/3.JPG',1)
img_nir = u.imgResize(img_nir,sh_scale)
d_blue = img_nir[:,:,1]
d_nir = img_nir[:,:,2]
#ndvi = getNDVI(img)

##leyendo NDVI

# obtener arreglo con nir


img = cv.imread('datos/ndvis/1.tif',cv.IMREAD_ANYDEPTH)
img = u.imgResize(img,sh_scale)

sh_scale =1;

if b_showing:
    u.imgShow(img, sh_scale, "ndvi")

#separar tierra de plantas, primera segmentación
k = 3;
max_iter = 5;
eps =.01;

u.imgDes(d_nir)

mask = s.segOtsu(img) #sotsu supone que hay dos cosas
u.imgDes(mask)
veg_zones_ndvi = cv.bitwise_and(img,img,mask = mask)
veg_zones_nir = cv.bitwise_and(d_nir,d_nir,mask = mask)
veg_zones_lv = cv.bitwise_and(d_blue,d_blue,mask = mask)

u.imgDes(veg_zones_ndvi)
veg_zones_nir_f32 = np.float32(veg_zones_nir);
veg_zones_lv_f32 = np.float32(veg_zones_lv);


#cv.imshow('veg_veg_zones', veg_zones_ndvi );
#cv.imshow('veg_veg_nir', veg_zones_nir );
#cv.imshow('veg_veg_lv', veg_zones_lv );

#imgF = cv.merge((veg_zones_nir_f32,veg_zones_nir_f32,veg_zones_ndvi));


rows, cols = veg_zones_ndvi.shape[:2];
p = np.zeros(( cols*rows,2));


p[:,0] = veg_zones_ndvi.reshape(( cols*rows,1))[:,0];

veg_zones_nir_f32 = veg_zones_nir_f32/255
p[:,1] = veg_zones_nir_f32.reshape(( cols*rows,1))[:,0];

#veg_zones_lv_f32 = veg_zones_lv_f32/255
#p[:,2] = veg_zones_lv_f32.reshape(( cols*rows,1))[:,0];


p = np.float32(p)


u.imgDes(p)
# p = np.float32(p)


x = s.segmetarByKMeans(veg_zones_ndvi,p,k,max_iter,eps)
#
cv.imshow('x', x );
#
# x = s.findCountorns(veg_zones,mask)
# cv.imshow('xs', x );

cv.waitKey(0);