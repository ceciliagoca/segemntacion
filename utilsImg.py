import numpy as np
import cv2 as cv

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

def imgDes(img):
    print ('type: ' , img.dtype , ' shape: ' , img.shape)
    return 0