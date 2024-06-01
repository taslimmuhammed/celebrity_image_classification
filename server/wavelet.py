import numpy as np
import pywt
import cv2

def w2d(img, mode='haar',level=1):
    imArray = img
    imArray = cv2.cvtColor(imArray, cv2.COLOR_RGB2GRAY)
    imArray =  np.float32(imArray)
    #normalisation
    imArray /= 255;
    coeff = pywt.wavedec2(imArray, mode, level=level)
    
    #Process Coefficients
    coeffs_H = list(coeff)
    coeffs_H[0] *=0;
    
    #reconstruction - inverse wavelet transformation
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *=255;
    imArray_H = np.uint8(imArray_H)
    
    return imArray_H