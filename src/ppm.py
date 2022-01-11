# -*- coding: utf-8 -*-
"""
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
############################  Pixels Per Metric Module  ##################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

This tool calculates a conversion ratio to translate pixels into a length unit.
This tool can optionally use color checker or the largest solid color square in the image.

This script requires that `OpenCV 2', 'numpy', and 'scipy' be installed within the Python environment you are running this script in.
This script imports the 'utility.py', module within the same folder.
"""

import numpy as np
import cv2
import argparse
import os
import utility
import sys
from scipy.spatial import distance as dist

def ppm_square(img, pixelspermetric):
    """Calculates the pixels per metric. 

    This tool allows the user to convert any 1D or 2D measurements from pixels to a know unit by providing a refference in the image.
    The refference must be a solid color square known dimensions. (Optional) Any square within a color checker may be used.

    Parameters
    ----------
    filename : array_like
        Valid file path to image. Accepted formats: 'tiff', 'jpeg', 'bmp', 'png'.
    pixelspermetric: float
        Refference length of largest square in image in any lenght unit of interest.

    Returns
    -------
    PixelsPerMetric
        Number of pixels per unit refference provided (centimeters, inches, etc.)
    ppm_proof
        Image showing the largest square and its ppm conversion ratio

    References
    ----------
    .. [1] Adrian Rosebrock, OpenCV, PyImageSearch, https://www.pyimagesearch.com/, accessed on 01 January 2020

    Thank you zbar! http://zbar.sourceforge.net/index.html

    Examples
    --------

    python ppm.py W201432.JPG 100

    """

    ppm_proof = img.copy()
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Find any objects with high saturation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _,_,v = cv2.split(hsv)
    _, sqr = cv2.threshold(v, 0, 255, cv2.THRESH_OTSU)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Find any roughly square objects
    cnts = cv2.findContours(sqr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    mask = np.zeros_like(sqr)
    for c in cnts:
        area_sqr = cv2.contourArea(c)
        #if 30000 > area_sqr > 500:
        if area_sqr > 500:    
            rects = cv2.minAreaRect(c)
            width_i = int(rects[1][0])
            height_i = int(rects[1][1])
            if height_i > width_i:
                rat = round(width_i/height_i, 2)
            else:
                rat = round(height_i/width_i, 2)
            if 0.95 <  rat < 1.1: 
                cv2.drawContours(mask, [c], -1, (255), -1)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Find and measure width largest square
    if cv2.countNonZero(mask) != 0:
        mask = utility.max_cnct(mask)
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for cs in cnts:         
            ppm_proof =cv2.drawContours(ppm_proof, [cs], -1, (53, 57, 250), 5)
            areas = cv2.contourArea(cs)
            print(areas)
            rects = cv2.minAreaRect(cs)
            boxs = cv2.boxPoints(rects)
            boxs = np.array(boxs, dtype="int")          
            boxs1 = utility.order_points(boxs)
            ppm_proof =cv2.drawContours(ppm_proof, [boxs.astype(int)], -1, (0, 255, 255), 10)
            (tls, trs, brs, bls) = boxs
            (tltrXs, tltrYs) = ((tls[0] + trs[0]) * 0.5, (tls[1] + trs[1]) * 0.5)
            (blbrXs, blbrYs) = ((bls[0] + brs[0]) * 0.5, (bls[1] + brs[1]) * 0.5)
            (tlblXs, tlblYs) = ((tls[0] + bls[0]) * 0.5, (tls[1] + bls[1]) * 0.5)
            (trbrXs, trbrYs) = ((trs[0] + brs[0]) * 0.5, (trs[1] + brs[1]) * 0.5)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~compute the Euclidean distance between the midpoints
            dBs = dist.euclidean((tlblXs, tlblYs), (trbrXs, trbrYs)) #pixel width
            PixelsPerMetric = dBs / pixelspermetric
            cv2.putText(ppm_proof, "{:.1f} Pixels per Metric".format(PixelsPerMetric),
                    (int(trbrXs), int(trbrYs - 180)), cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 0, 255), 15)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~draw midpoints and lines on proof
            cv2.line(ppm_proof, (int(tlblXs), int(tlblYs)), (int(trbrXs), int(trbrYs)), (255, 0, 255), 20) #width
            cv2.circle(ppm_proof, (int(tlblXs), int(tlblYs)), 23, (255, 0, 255), -1) #left midpoint
            cv2.circle(ppm_proof, (int(trbrXs), int(trbrYs)), 23, (255, 0, 255), -1) #right midpoint
    else:
        PixelsPerMetric = ppm_proof = None
    
    return PixelsPerMetric, ppm_proof

if __name__ == "__main__":
    print("You are running ppm.py solo...")
    
    filename = sys.argv[1]                          # Translating arguments into something the function above can understand
    img=cv2.imread(filename)
    
    reff_len = float(sys.argv[2])
        
    PixelsPerMetric, ppm_proof = ppm_square(img, reff_len)   #Run the pixels per metric module without color checker  

    print("[PPM]--{}--Found {} pixels per metric".format(filename, PixelsPerMetric))
    