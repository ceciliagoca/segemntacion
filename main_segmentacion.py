__author__ = "cecilia"
__date__ = "$Feb 23, 2016 11:05:50 AM$"
__update__ = "$Mar 03, 2016 11:05:50 AM$"

import numpy as np
import cv2 as cv
import utilsImg as u
import segmentacion as s
import sklearn.cluster as sc


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


def showClustersRes(objetos,y,img,wname='resCluster'):
    i=0;
    for x in objetos:
        roi = x.imagen;
        if y[i]==1:
            roi [x.mascara != 0 ] = 100;
        i=i+1;

        rows, cols, channels =img.shape
        mask = x.mascara #ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        # mask_inv = cv.bitwise_not(mask)

        mask1 = np.zeros((rows,cols),np.uint8)
        mask1[x.rect_y:x.rect_y+x.rect_h, x.rect_x:x.rect_x+x.rect_w] = mask[0:x.rect_h, 0:x.rect_w ]
        img2_fg = cv.bitwise_and(img, img, mask=mask1)

        # Put logo in ROI and modify the main image
        dst = cv.add(img, img2_fg)
        img[0:rows, 0:cols] = dst

    cv.imshow(wname,img)
    cv.waitKey(0);



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
    cv.imshow('sss',original)


    #Parametros para diferentes resolusiones

    k_otsu = 11 if res==1 else  5;
    k_omorf = 6  if res==1 else 3;


    # Segmentar tierra y vegetación
    veg_mask = s.segOtsu(original, k_otsu)
    #cv.imshow('veg_mask', veg_mask)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k_omorf, k_omorf))
    veg_mask = cv.morphologyEx(veg_mask, cv.MORPH_CLOSE, kernel, iterations=1)
    #cv.imshow('veg_mask2', veg_mask)

    # create a CLAHE object (Arguments are optional).
    #clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #img_e = clahe.apply(original)
    img_e = cv.equalizeHist(original)
    img_veg = cv.bitwise_and(img_e, img_e, mask=veg_mask)
    cv.imshow('img_veg', img_veg)
    #cv.waitKey()

    ##crear semilas
    dist = cv.distanceTransform(veg_mask , cv.DIST_L2, 3)  # distanceTransform(bw, dist, CV_DIST_L2, 3);
    cv.normalize(dist, dist, 0, 1., cv.NORM_MINMAX);
    dist = dist * 255
    dist = np.uint8(dist)
    #cv.imshow('seem0', dist)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k_omorf*2, k_omorf*2))
    seem = cv.morphologyEx(dist, cv.MORPH_ERODE, kernel, iterations=4)
    seem = cv.morphologyEx(seem, cv.MORPH_CLOSE, kernel, iterations=4)
    cv.imshow('seem1', seem)
    #cv.waitKey(0)

    # apilcar watershe
    markes1, w_img =s.watershed2(seem, img_veg,'w_sege')
    s.watershed2(seem, original, 'w_completa')

    markes2 = np.zeros(markes1.shape)
    markes2[markes1!=0]=255;
    #markes1[markes1!=0] =0;
    cv.imshow('water markers', w_img )

    w_img = cv.morphologyEx(w_img, cv.MORPH_ERODE, kernel, iterations=1)
    w_img = cv.morphologyEx(w_img, cv.MORPH_CLOSE, kernel, iterations=2)


    canny = cv.Canny(w_img, 100, 130,9, 5);
    # cv.imshow('markers', canny)
    # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2,2))
    # canny = cv.morphologyEx(canny, cv.MORPH_ERODE, kernel, iterations=1)
    # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5 , 5))
    # canny = cv.morphologyEx(canny, cv.MORPH_DILATE, kernel, iterations=1)
    # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    # canny = cv.morphologyEx(canny, cv.MORPH_ERODE, kernel, iterations=2)
    # cv.imshow('markers', canny)

    im2, contours, hierarchy = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    #img = cv.DrawContours(w_img, contours, -1, (255, 0, 0),thickness=cv.CV_FILLED)
    img = cv.drawContours(w_img, contours,  -1 ,(100, 0, 0),4)

    cv.imshow('contorn3os', img)

    i=0
    lechuga = []
    #boxing countours
    for countorn in contours:
        #R = cv.boundingRect(countorn);
        x, y, w, h = cv.boundingRect(countorn)

        #rect = cv.minAreaRect(countorn)
        #box = cv.boxPoints(rect)
        #box = np.int0(box)
        #cv.drawContours(img, [box], 0, (0, 0, 255), 2)
        perimeter = cv.arcLength(countorn, True)
        if perimeter > 150 :
            cc = original[y:y+h, x:x+w]
            lechuga.append(cc)
            i=i+1
            hist = cv.calcHist([img], [0], None, [256], [0, 256])
            cv.imshow('contornos', cc)

            cv.waitKey();


    cv.imshow('contornos', img)
    cv.waitKey();

    return ;

def segmentacion_06(img,res):
    # lectura de imagen
    if img.dtype != np.uint8:
        img = img * 255
        original = np.uint8(img)
    else:
        original = img
    cv.imshow('imagen', original)



    # Parametros para diferentes resolusiones

    k_otsu = 11 if res == 1 else  5;
    k_omorf = 6 if res == 1 else 3;

    # Segmentar tierra y vegetación
    veg_mask = s.segOtsu(original, k_otsu)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k_omorf, k_omorf))
    veg_mask = cv.morphologyEx(veg_mask, cv.MORPH_CLOSE, kernel, iterations=1)
    img_e = cv.equalizeHist(original)
    img_veg = cv.bitwise_and(img_e, img_e, mask=veg_mask)
    cv.imshow('img_veg', img_veg)

    seeds = s.findSeeds(img_veg,k_omorf)

    m_watershed = s.watershed2(seeds, img_veg)



    return m_watershed




if __name__ == "__main__":
      print ("Segmentacion de lechugas")
      print ("version de opencv: " + cv.__version__)


img_nir = cv.imread('datos/lechuga_ndvi/4.JPG',1)
img_nir = u.imgResize(img_nir,sh_scale)
d_nir = img_nir[:,:,2]

markers = segmentacion_06(img_nir[:,:,2], res=1) # 1 <<- res : resolucion alta ; 0 resolucion de vuelo
#s.showMarkes(markers, 'd');
objetos = s.getObjects(img_nir[:,:,2],markers)


c_kmeans =  sc.KMeans(n_clusters=2)
h_objetos = np.zeros((len(objetos),8),np.float64)
i=0;
for x in objetos:

    hist_t = np.float64(x.histograma.T)
    hist_t = cv.normalize(hist_t,hist_t)
    h_objetos[i:]=hist_t;
    i=i+1;

c_kmeans.fit(h_objetos)
y = c_kmeans.predict(h_objetos)


objetosAP = s.getObjects(img_nir[:,:,2],markers)
c_AP =  sc.AffinityPropagation()
h_objetos = np.zeros((len(objetosAP),8),np.float64)
i=0;
for x in objetos:

    hist_t = np.float64(x.histograma.T)
    hist_t = cv.normalize(hist_t,hist_t)
    h_objetos[i:]=hist_t;
    i=i+1;

c_AP.fit(h_objetos)
yAP = c_AP.predict(h_objetos)

#print resultados
showClustersRes(objetos,yAP,img_nir,'AP')
showClustersRes(objetos,y,img_nir,'Kmeans')



cv.waitKey(0)