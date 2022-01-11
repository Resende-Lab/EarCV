# -*- coding: utf-8 -*-
"""
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
##################################  Color Correction Module  #############################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

This tool corrects the color of an image that contains a color checker.

This script requires that `OpenCV 2', 'numpy', and 'plantcv' be installed within the Python environment you are running this script in.
This script imports the 'utility.py' module within the same folder.
"""

import numpy as np
import cv2
from plantcv import plantcv as pcv
import argparse
import os
import utility
import sys

def color_correct(filename, img, reff, debug):
    """Corrects the color of an image that contains a color checker based on reference.

    This tool can optionally use any reference image of a color checker for color correction.
    You may use the provided reference in this package called clrchr.png.
    Credit to: Nayanika Ghosh, https://github.com/juang0nzal3z/EarCV/tree/main/ColorHomography

    Parameters
    ----------
    filename : array_like
        Valid file path to image to be color corrected. Accepted formats: 'tiff', 'jpeg', 'bmp', 'png'.
    
    reff_name: array-like
        Valid file path to reference image to be used as ground truth for color correction. Accepted formats: 'tiff', 'jpeg', 'bmp', 'png'.

    debug: bool
        If true, print output proof images.
    
    Returns
    -------
    tar_chk: 
        Image: color checker mask from original image to to corrected
    corrected: 
        Image: Image after color checker correction
    avg_tar_error: 
        Int. RMS error of original color checker
    avg_trans_error:
        Int. RMS error after color correction
    csv_field:
        Vector of 26 values containing correction metrics to asses performance:
        'Filename', 'Overall improvement', 'Square1', 'Square1', 'Square3', 'Square4', 'Square5', 'Square6',
        'Square7', 'Square8', 'Square9', 'Square10', 'Square11', 'Square12', 'Square13', 'Square14',
        'Square15', 'Square16', 'Square17', 'Square18', 'Square19', 'Square20', 'Square21', 'Square22', 'Square23', 'Square24'

    References
    ----------
    .. [1] Algorithm based on: https://homepages.inf.ed.ac.uk/rbf/PAPERS/hgcic16.pdf

    Examples
    --------

    python ppm.py W201432.JPG 100


    """

def color_correct(filename, img, reff, debug):

    if reff is not None:
        src_chk = reff
        srcImg = img
        # src_chk is the extracted source color checker image
        #src_chk = srcImg

        # Extract the color chip mask and RGB color matrix from source color checker image
        dataframe1, start, space = pcv.transform.find_color_card(rgb_img=src_chk, background='dark')
        src_mask = pcv.transform.create_color_card_mask(rgb_img=src_chk, radius=5, start_coord=start, spacing=space,
                                                    ncols=4, nrows=6)
        src_head, src_matrix = pcv.transform.get_color_matrix(src_chk, src_mask)
        S = np.zeros((np.shape(src_matrix)[0], 3))
        for r in range(0, np.ma.size(src_matrix, 0)):
            S[r][0] = src_matrix[r][1]
            S[r][1] = src_matrix[r][2]
            S[r][2] = src_matrix[r][3]
        S_reshaped = np.reshape(S, (6, 4, 3))

    else:
        # Use hard coded reference if no reference image is provided
        src_matrix = np.array([[10, 170, 189, 103], [20, 46, 163,224], [30, 161, 133, 8], [40, 52, 52, 52],
                      [50, 177, 128, 133], [60, 64, 188, 157], [70, 149, 86, 187], [80, 85, 85, 85],
                      [90, 67,108,87], [100, 108,60,94], [110, 31, 199, 231], [120, 121, 122, 122],
                      [130, 157, 122, 98], [140, 99, 90, 193], [150, 60, 54, 175], [160, 160,160,160],
                      [170, 130, 150, 194], [180, 166, 91, 80], [190, 73, 148, 70], [200, 200, 200, 200],
                      [210, 68, 82, 115], [220, 44, 126, 214], [230, 150, 61, 56], [240, 242, 243, 243]])
        S = np.array([[170, 189, 103], [46, 163,224], [161, 133, 8], [52, 52, 52],
                      [177, 128, 133], [64, 188, 157], [149, 86, 187], [85, 85, 85],
                      [67,108,87], [108,60,94], [31, 199, 231], [121, 122, 122],
                      [157, 122, 98], [99, 90, 193], [60, 54, 175], [160,160,160],
                      [130, 150, 194], [166, 91, 80], [73, 148, 70], [200, 200, 200],
                      [68, 82, 115], [44, 126, 214], [150, 61, 56], [242, 243, 243]])
        S_reshaped = np.reshape(S,(6, 4, 3))

    # TARGET IMAGE
    tarImg = img
    # Extract the color chip mask and RGB color matrix from target color checker image
    y, tar_chk = utility.clr_chk(tarImg)

    dataframe1, start, space = pcv.transform.find_color_card(rgb_img=tar_chk, background='dark')
    tar_mask = pcv.transform.create_color_card_mask(rgb_img=tar_chk, radius=5, start_coord=start, spacing=space,
                                                ncols=4, nrows=6)
    tar_head, tar_matrix = pcv.transform.get_color_matrix(tar_chk, tar_mask)
    T = np.zeros((np.shape(tar_matrix)[0], 3))
    for r in range(0, np.ma.size(tar_matrix, 0)):
        T[r][0] = tar_matrix[r][1]
        T[r][1] = tar_matrix[r][2]
        T[r][2] = tar_matrix[r][3]
    T_reshaped = np.reshape(T, (6, 4, 3))
    if utility.dist_rgb(S_reshaped[0][3], T_reshaped[0][3]) > utility.dist_rgb(S_reshaped[0][3], T_reshaped[5][0]):
        T_reshaped = np.rot90(T_reshaped, axes=(0, 1))
        T_reshaped = np.rot90(T_reshaped, axes=(0, 1))
        T = np.reshape(T_reshaped, (24, 3))

    # Call functions from ColorHomo to generate and apply the color homography matrix
    homography = utility.generate_homography(S, T)
    corr = utility.apply_homo(tar_chk, homography, False)
    corrected = utility.apply_homo(tarImg, homography, True)

    (avg_tar_error, avg_trans_error, csv_field) = utility.calculate_color_diff(filename, tarImg, src_matrix, tar_matrix, corr)
    #corrected = cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR)
    #tarImg = cv2.cvtColor(tarImg, cv2.COLOR_RGB2BGR)
    #srcToShow = cv2.cvtColor(srcImg, cv2.COLOR_RGB2BGR)

    (avg_tar_error, avg_trans_error, csv_field) = utility.calculate_color_diff(filename, tarImg, src_matrix, tar_matrix, corr)
    
    color_proof = cv2.vconcat([tarImg, corrected])
    #corrected = cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR)
    #tarImg = cv2.cvtColor(tarImg, cv2.COLOR_RGB2BGR)
    #srcToShow = cv2.cvtColor(src_chk, cv2.COLOR_RGB2BGR)
    if debug is True:
        #utility.plot_images(srcToShow, tarImg, corrected)          # This is Nyanika's visual output
        cv2.namedWindow('[DEBUG] [COLOR] Color Correction Proof', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('[DEBUG] [COLOR] Color Correction Proof', 1000, 1000)
        cv2.imshow('[DEBUG] [COLOR] Color Correction Proof', color_proof); cv2.waitKey(3000); cv2.destroyAllWindows()

    return color_proof, tar_chk, corrected, avg_tar_error, avg_trans_error, csv_field


if __name__ == "__main__":
    print("You are running ColorCorrection.py solo...")
    
    filename = sys.argv[1]                          # Translating arguments into something the function above can understand
    img=cv2.imread(filename)
    
    reff_name = sys.argv[2]
    reff=cv2.imread(reff_name)
    
    debug = sys.argv[3]
    if debug == "True":
        debug = True
    else:
        debug = False
    

    color_proof, tarImg, corrected, avg_tar_error, avg_trans_error, csv_field = color_correct(filename, img, reff, debug)
        
    print("[COLOR]--{}--Before correction - {} After correction - {}".format(filename, avg_tar_error, avg_trans_error))


    #filename = sys.argv[1]                          # Translating arguments into something the function above can understand
    #img=cv2.imread(filename)
    #qr_window_size = sys.argv[2]
    #QRcodeType, QRcodeData, QRcodeRect, qr_count, qr_proof = qr_scan(img, qr_window_size, overlap, debug) # Run the qr.py module
    #print("[QR]--{}--Found {}: {} on the {}th iteration".format(filename, QRcodeType, QRcodeData, qr_count))    # Log
