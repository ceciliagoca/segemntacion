import numpy as np
import cv2 as cv
import utilsImg as u

def segmetarByKMeans(img,p, k, max_iter, eps):

    # rows, cols = img.shape[:2];
    # p = img.reshape(( cols*rows,1));
    # p = np.float32(p)

    #attempts  Flag to specify the number of times the algorithm is executed using different initial labellings.
    attempts =10;
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, max_iter, eps)
    ret,label,center=cv.kmeans(p,k,None,criteria,attempts,cv.KMEANS_PP_CENTERS)
    center2 =  center[:,0]
    a = sorted(range(len(center2)), key=lambda k:center2[k]) #indices oordenados de menor a mayor

    b = 255 / (k-1);
    aux = label.copy()

    for x in range(0,k):
        aux[label == a[x]] = (x*b)
        print (x)

    aux = aux.reshape(img.shape)
    img_seg = np.uint8(aux)

    return img_seg

def findCountorns(img, img_bin): #recibe imagen de tres canales de grises y binaria con formato +uint8

    #verificar tipos de datos
    cv.imshow('img_bin2', img_bin)
    cv.waitKey(0);
    im2, contours, hierarchy = cv.findContours(img_bin,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    print (len(contours))
    cv.drawContours(img, contours, -1, (255,0,0), 6)
    ##wathershed

    #
    # ret, markers = cv.connectedComponents(img_bin)
    # markers = markers+1
    # #markers[unknown ==255] = 0
    # markers = cv.watershed(img,markers)
    #
    # rows, cols = img.shape[:2];
    # img2 = np.zeros((rows,cols,3),np.uint8)
    # img2[markers == -1] = [255,0,0]
    #
    #
    #
    # cv.imshow('treshh3', img2)
    # cv.waitKey(0)


    return img

def segmentar(img):

    rows, cols = img.shape[:2];
    cv.imshow('maskeddata', img)
    cv.waitKey(0)

    opening = cv.morphologyEx(img, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5)))
    opening = cv.morphologyEx(opening, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE,(11,11)))

    kernel = np.ones((7,7),np.uint8)
    erosion = cv.erode(opening,cv.getStructuringElement(cv.MORPH_ELLIPSE,(7,7)),iterations = 2);
    ret, thresh = cv.threshold(erosion,.5,1 ,cv.THRESH_BINARY)
    cv.imshow('erosion', erosion)
    cv.waitKey(0)

    x2 = np.zeros((rows,cols,3), np.uint8);

    myeros = erosion*256;
    thresh[thresh!=0]=255;

    x = np.uint8(thresh)

    cv.imshow('cccon', x)
    cv.waitKey(0)


    xmyeros= np.uint8(myeros)
    print (xmyeros.dtype , ' ' , xmyeros.shape)
    print (x2.dtype , ' ' , x2.shape)
    x2 =  cv.cvtColor(xmyeros,cv.COLOR_GRAY2RGB)

    aux = findCountorns(x2,x)
    cv.imshow('con', aux)
    cv.waitKey(0)
    #watershed(erosion)

    return aux

def watershed(img):
    print (img.shape)
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

def segOtsu(img): #return a mask

    img = img*255
    img = np.uint8(img)

    img = cv.medianBlur(img,5)

    # th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,2)
    # th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
    #
    #
    # cv.imshow('img1',img )
    # cv.imshow('th1',th1 )
    # cv.imshow('th2',th2 )
    # cv.imshow('th3', th3)
    # cv.waitKey(0);
    #
    # cv.destroyAllWindows()
    #
    #
    #
    # # global thresholding
    # ret1,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
    # # Otsu's thresholding
    # ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    # # Otsu's thresholding after Gaussian filtering

    #blur = cv.GaussianBlur(img,(1,1),0)
    ret3,mask = cv.threshold(img,100,255,cv.THRESH_BINARY+cv.THRESH_OTSU)


    #cv.imshow('img1',img )
    #cv.imshow('th3', th3)
    #cv.waitKey(0);

    return mask


