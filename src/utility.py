""" --------------------------------------------------------------------------------------------------
 This file has all the utility(helper) functions defined
 ----------------------------------------------------------------------------------------------------"""

import argparse
import logging
import sys
import traceback
import os
import re
import string
import csv
import cv2
import math
import numpy as np
from statistics import stdev, mean
from matplotlib import pyplot as plt
from pyzbar.pyzbar import decode 
from scipy.spatial import distance as dist
from plantcv import plantcv as pcv
from scipy import sparse



""" --------------------------------------------------------------------------------------------------
 This function calculates the RMS difference between the mean RGB values of source and target color checker chip
 It also writes the overall % improvement in reduction of RMS difference as well as for each of the 24 color
  chip to a csv file
 ----------------------------------------------------------------------------------------------------"""

def calculate_color_diff(filename,target_im, src_matrix, tar_matrix, transfer_chk):
    # RGB value based difference between source and target image
    dist = []
    dist_r = []
    dist_g = []
    dist_b = []
    avg_src_error = 0.0
    for r in range(0, np.ma.size(tar_matrix, 0)):
        for i in range(0, np.ma.size(src_matrix, 0)):
            if tar_matrix[r][0] == src_matrix[i][0]:
                r_mean = math.pow((tar_matrix[r][1] - src_matrix[i][1]), 2)
                g_mean = math.pow((tar_matrix[r][2] - src_matrix[i][2]), 2)
                b_mean = math.pow((tar_matrix[r][3] - src_matrix[i][3]), 2)
                dist_r.append(math.sqrt(r_mean))
                dist_g.append(math.sqrt(g_mean))
                dist_b.append(math.sqrt(b_mean))
                temp = math.sqrt((r_mean + g_mean + b_mean)/3)
                avg_src_error = avg_src_error + temp
                dist.append(temp)
    avg_src_error /= np.ma.size(tar_matrix, 0)

    # Corrected image
    # Extract the color chip mask and RGB color matrix from transfer color checker image
    dataframe1, start, space = pcv.transform.find_color_card(rgb_img=transfer_chk, background='dark')
    transfer_mask = pcv.transform.create_color_card_mask(rgb_img=transfer_chk, radius=5, start_coord=start,
                                                         spacing=space,
                                                         ncols=4, nrows=6)
    transfer_head, transfer_matrix = pcv.transform.get_color_matrix(transfer_chk, transfer_mask)

    # RGB value based difference between source and transfer image
    dist1 = []
    dist1_r = []
    dist1_g = []
    dist1_b = []
    csv_field = [target_im]
    avg_transfer_error = 0.0
    for r in range(0, np.ma.size(transfer_matrix, 0)):
        for i in range(0, np.ma.size(src_matrix, 0)):
            if transfer_matrix[r][0] == src_matrix[i][0]:
                r1_mean = math.pow((transfer_matrix[r][1] - src_matrix[i][1]), 2)
                g1_mean = math.pow((transfer_matrix[r][2] - src_matrix[i][2]), 2)
                b1_mean = math.pow((transfer_matrix[r][3] - src_matrix[i][3]), 2)
                dist1_r.append(math.sqrt(r1_mean))
                dist1_g.append(math.sqrt(g1_mean))
                dist1_b.append(math.sqrt(b1_mean))
                temp = math.sqrt((r1_mean + g1_mean + b1_mean)/3)
                avg_transfer_error = avg_transfer_error + temp
                dist1.append(temp)
                csv_field.append((dist[i]-temp)/float(dist[i])*100)
    avg_transfer_error /= np.ma.size(tar_matrix, 0)
    csv_field.insert(1, ((avg_src_error-avg_transfer_error)/float(avg_src_error))*100)
    return (avg_src_error, avg_transfer_error, csv_field)

""" --------------------------------------------------------------------------------------------------
 Learn the color homography matrix between two color checker images using alternating least square method
 ----------------------------------------------------------------------------------------------------"""

def generate_homography(src_img, tar_img):
    max_iter = 100
    tol = 1e-10
    (H, D) = calculate_H_using_ALS(tar_img, src_img, max_iter, tol)
    return H


def calculate_H_using_ALS(p1, p2, max_iter, tol):
    Npx = len(p1)    # Num of data
    N = p1
    D = sparse.eye(Npx, Npx)
    n_it = 0
    ind1 = np.sum((p1 > 0) & (p1 < np.Inf), 0) == Npx
    ind2 = np.sum((p2 > 0) & (p2 < np.Inf), 0) == Npx
    vind = ind1 & ind2
    print(vind)
    # TODO: Add a size check for p1 & p2
    while(n_it < max_iter):
        n_it = n_it + 1
        D = solve_D(N, p2)
        P_D = np.dot(D, p1)
        P_X = np.linalg.pinv(P_D[:, :])
        H = np.dot(P_X, p2[:, :])
        N = np.dot(P_D, H)
    PD = D
    return H, PD


def solve_D(p, q):
    nPx = len(p)
    nCh = len(p[0])
    d = np.divide(np.dot(np.ones((1, nCh)), np.transpose(p*q)), np.dot(np.ones((1, nCh)), np.transpose(p*p)))
    D = sparse.spdiags(d, 0, nPx, nPx)
    D = sparse.dia_matrix.astype(D, float).toarray()
    return D

""" --------------------------------------------------------------------------------------------------
 Apply the learnt color homography matrix to the target image
 Reference : https://homepages.inf.ed.ac.uk/rbf/PAPERS/hgcic16.pdf
 ----------------------------------------------------------------------------------------------------"""

def apply_homo(tar, cor_mat, isTarImage):
    img_size = np.shape(tar)
    rgb_tar = np.reshape(tar, [img_size[0]*img_size[1],3])

    if isTarImage:
        corrected = rgb_tar
        corrected = np.dot(corrected, cor_mat)
        corrected = np.reshape(corrected, img_size)
    else:
        corrected = np.dot(rgb_tar, cor_mat)
        corrected = np.reshape(corrected, img_size)
    return corrected.astype(np.uint8)


""" --------------------------------------------------------------------------------------------------
 Helper function to find coordinates to split image into overlapping subblocks (used in QR scan)
 ----------------------------------------------------------------------------------------------------"""

def start_points(size, split_size, overlap=10):
	points = [0]
	stride = int(split_size * (1-overlap))
	counter = 1
	while True:
		pt = stride * counter
		if pt + split_size >= size:
			points.append(size - split_size)
			break
		else:
			points.append(pt)
		counter += 1
	return points

""" --------------------------------------------------------------------------------------------------
 Parses input path into root, filename, and extension
 ----------------------------------------------------------------------------------------------------"""

def img_parse(fullpath):
	fullpath = fullpath
	root_ext = os.path.splitext(fullpath) 
	ext = root_ext[1]											
	filename = root_ext[0]										#File  ID
	try:
		root = filename[:filename.rindex('/')+1]
	except:
		root = "./"
	try:
		filename = filename[filename.rindex('/')+1:]
	except:
		filename = filename
	return fullpath, root, filename, ext	

''' --------------------------------------------------------------------------------------------------
 Order box coordinates from top left, going clockwise, to bottom left
 Sort w.r.t x-coordinate first, retrieve top left & bottom right from it
 Then do similar with the y-coordinate
 ----------------------------------------------------------------------------------------------------'''
def order_points_new(pts):
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    left = x_sorted[:2, :]
    right = x_sorted[2:, :]
    left = left[np.argsort(left[:, 1]), :]
    (top_left, bottom_left) = left
    right = right[np.argsort(right[:, 1]), :]
    (top_right, bottom_right) = right
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")


''' --------------------------------------------------------------------------------------------------
 Calculate euclidean distance between two point coordinates
 ----------------------------------------------------------------------------------------------------'''
def calc_distance(x, y):
    dis = round(math.sqrt(math.pow(x[0]-y[0], 2) + math.pow(x[1]-y[1], 2)))
    return dis

''' --------------------------------------------------------------------------------------------------
 # Cut image into N slices
 ----------------------------------------------------------------------------------------------------'''
def ranges(N, nb):
	step = N / nb
	return ["{},{}".format(round(step*i), round(step*(i+1))) for i in range(nb)]


''' --------------------------------------------------------------------------------------------------
 Calculate the correctly oriented coordinates of checkerboard
 ----------------------------------------------------------------------------------------------------'''
def get_dest_coord(boxs):
    fsc = []
    sc = []
    tc = []
    frc = []
    if calc_distance(boxs[0], boxs[3]) < calc_distance(boxs[0], boxs[1]):
        fsc.append(boxs[3][0])
        fsc.append(boxs[3][1])
        sc.append(boxs[3][0])
        sc.append(boxs[3][1] - calc_distance(boxs[0], boxs[1]))
        tc.append(boxs[3][0] + calc_distance(boxs[0], boxs[3]))
        tc.append(boxs[3][1] - calc_distance(boxs[0], boxs[1]))
        frc.append(boxs[3][0] + calc_distance(boxs[0], boxs[3]))
        frc.append(boxs[3][1])
    else:
        fsc.append(boxs[0][0])
        fsc.append(boxs[0][1])
        sc.append(boxs[0][0] + calc_distance(boxs[0], boxs[1]))
        sc.append(boxs[0][1])
        tc.append(boxs[0][0] + calc_distance(boxs[0], boxs[1]))
        tc.append(boxs[0][1] + calc_distance(boxs[0], boxs[3]))
        frc.append(boxs[0][0])
        frc.append(boxs[0][1] + calc_distance(boxs[0], boxs[3]))
    return np.array([fsc, sc, tc, frc])


''' --------------------------------------------------------------------------------------------------
 Calculate euclidean distance between two RGB colors
 ----------------------------------------------------------------------------------------------------'''
def dist_rgb(mat1, mat2):
    dis = round(math.sqrt(math.pow(mat1[0]-mat2[0], 2) + math.pow(mat1[1]-mat2[1], 2)) + math.pow(mat1[2]-mat2[2], 2))
    return dis


''' --------------------------------------------------------------------------------------------------
 Extract the checkerboard from the image after orienting it in the standard way
 ----------------------------------------------------------------------------------------------------'''
def clr_chk(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)  # Split into it channel constituents
    clr_ck = cv2.threshold(v, 65, 256, cv2.THRESH_BINARY)[1]  # Threshold
    clr_ck = cv2.morphologyEx(clr_ck, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=6)
    '''cv2.namedWindow('Reff', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Reff', 1000, 1000)
    cv2.imshow('Reff', clr_ck);
    cv2.waitKey(10000);'''
    cnts = cv2.findContours(clr_ck, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE);
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    clr_ck = np.zeros_like(s)
    for c in cnts:
        area_tip = cv2.contourArea(c, oriented=0)
        if 2000 < area_tip < ((img.shape[0] * img.shape[1]) * 0.05):
            rects = cv2.minAreaRect(c)
            boxs = cv2.boxPoints(rects)
            boxs = np.array(boxs, dtype="int")
            width_i = int(rects[1][0])
            height_i = int(rects[1][1])
            if height_i > width_i:
                rat = round(width_i / height_i, 2)
            else:
                rat = round(height_i / width_i, 2)
            if rat > 0.96:
                cv2.drawContours(clr_ck, [c], -1, (255), -1)
    clr_ck = cv2.morphologyEx(clr_ck, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)),
                              iterations=10)
    _,cnts,_ = cv2.findContours(clr_ck, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE);
    for c in cnts:
        rects = cv2.minAreaRect(c)
        boxs = cv2.boxPoints(rects)
        boxs = np.array(boxs, dtype="int")
        boxs = order_points_new(boxs)
        pts_dst = get_dest_coord(boxs)
        h, status = cv2.findHomography(boxs, pts_dst)
        img = cv2.warpPerspective(img, h, (img.shape[1], img.shape[0]))
    # repeat
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)  # Split into it channel constituents
    clr_ck = cv2.threshold(s, 65, 256, cv2.THRESH_BINARY)[1]  # Threshold
    clr_ck = cv2.morphologyEx(clr_ck, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=6)
    cnts = cv2.findContours(clr_ck, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE);
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    clr_ck = np.zeros_like(s)
    for c in cnts:
        area_tip = cv2.contourArea(c, oriented=0)
        if 2000 < area_tip < ((img.shape[0] * img.shape[1]) * 0.05):
            rects = cv2.minAreaRect(c)
            width_i = int(rects[1][0])
            height_i = int(rects[1][1])
            if height_i > width_i:
                rat = round(width_i / height_i, 2)
            else:
                rat = round(height_i / width_i, 2)
            if rat > 0.94:
                cv2.drawContours(clr_ck, [c], -1, (255), -1)
    # cv2.namedWindow('Reff', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Reff', 1000, 1000)
    # cv2.imshow('Reff', mask); cv2.waitKey(5000); cv2.destroyAllWindows()
    clr_ck = cv2.morphologyEx(clr_ck, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)),
                              iterations=10)
    # take a binary image and run a connected component analysis
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(clr_ck, connectivity=8)
    # extracts sizes vector for each connected component
    sizes = stats[:, -1]
    # initiate counters
    max_label = 1
    max_size = sizes[1]
    # loop through and fine the largest connected component
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]
    # create an empty array and fill only with the largest the connected component
    clr_ck = np.zeros(clr_ck.shape, np.uint8)
    clr_ck[output == max_label] = 255
    # return a binary image with only the largest connected component
    cnts = cv2.findContours(clr_ck, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE);
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    clr_ck = np.zeros_like(s)
    for c in cnts:
        rects = cv2.minAreaRect(c)
        boxs = cv2.boxPoints(rects)
        boxs = np.array(boxs, dtype="int")
        start_point = (boxs[1][0], boxs[1][1])
        end_point = (boxs[3][0] + 180, (boxs[3][1]))
        clr_ck = cv2.rectangle(clr_ck, start_point, end_point, (255), -1)
    clr_ck = cv2.dilate(clr_ck, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)), iterations=2)
    img_chk = img.copy()
    img_chk[clr_ck == 0] = 0
    return [img,img_chk]


""" -----------------------------------------------------------------------------------------------------
 Plot images before and after correction next to each other along with reference
 -----------------------------------------------------------------------------------------------------"""
def plot_images(src, tar, corrected):
    plt.subplot(2, 2, 1)
    plt.imshow(src)
    plt.title("Reference color checker")
    plt.subplot(2, 2, 3)
    plt.imshow(tar)
    plt.title("Before Correction")
    plt.subplot(2, 2, 4)
    plt.imshow(corrected)
    plt.title("After Correction")
    plt.show()


""" -----------------------------------------------------------------------------------------------------
Returns largest connected component
 -----------------------------------------------------------------------------------------------------"""
def max_cnct(binary):
# take a binary image and run a connected component analysis
	nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
# extracts sizes vector for each connected component
	sizes = stats[:, -1]
#initiate counters
	max_label = 1
	max_size = sizes[1]
#loop through and fine the largest connected component
	for i in range(2, nb_components):
		if sizes[i] > max_size:
			max_label = i
			max_size = sizes[i]
#create an empty array and fill only with the largest the connected component
	cnct = np.zeros(binary.shape, np.uint8)
	cnct[output == max_label] = 255
#return a binary image with only the largest connected component
	return cnct										# Returns largest connected component

""" -----------------------------------------------------------------------------------------------------
Returns obejct the coordinates in top-left order
 -----------------------------------------------------------------------------------------------------"""
def order_points(pts):
	# sort the points based on their x-coordinates
	xSorted = pts[np.argsort(pts[:, 0]), :]
	# grab the left-most and right-most points from the sorted
	# x-roodinate points
	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]
	# now, sort the left-most coordinates according to their
	# y-coordinates so we can grab the top-left and bottom-left
	# points, respectively
	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(tl, bl) = leftMost
	# now that we have the top-left coordinate, use it as an
	# anchor to calculate the Euclidean distance between the
	# top-left and right-most points; by the Pythagorean
	# theorem, the point with the largest distance will be
	# our bottom-right point
	D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
	(br, tr) = rightMost[np.argsort(D)[::-1], :]
	# return the coordinates in top-left, top-right,
	# bottom-right, and bottom-left order
	return np.array([tl, tr, br, bl], dtype="float32")

""" -----------------------------------------------------------------------------------------------------
Return a binary image with only the largest connected component, filled
 -----------------------------------------------------------------------------------------------------"""
def cnctfill(binary):
	# take a binary image and run a connected component analysis
	nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
	# extracts sizes vector for each connected component
	sizes = stats[:, -1]
	#initiate counters
	max_label = 1
	if len(sizes) > 1:
		max_size = sizes[1]
	#loop through and fine the largest connected component
		for i in range(2, nb_components):
			if sizes[i] > max_size:
				max_label = i
				max_size = sizes[i]
	#create an empty array and fill only with the largest the connected component
		cnct = np.zeros(binary.shape, np.uint8)
		cnct[output == max_label] = 255
	#take that connected component and invert it
		nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(cv2.bitwise_not(cnct), connectivity=8)
	
        

    # extracts sizes vector for each connected component
		sizes = stats[:, -1]
	#initiate counters
		max_label = 1
		max_size = sizes[1]
	#loop through and fine the largest connected component
		for i in range(2, nb_components):
			if sizes[i] > max_size:
				max_label = i
				max_size = sizes[i]
	#create an empty array and fill only with the largest the connected component
		filld = np.zeros(binary.shape, np.uint8)
		filld[output == max_label] = 255
		filld = cv2.bitwise_not(filld)
	else:
		filld = binary
	#return a binary image with only the largest connected component, filled
	return filld

""" -----------------------------------------------------------------------------------------------------
Change contrast of image
 -----------------------------------------------------------------------------------------------------"""
def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
 
 	if brightness != 0:
 		if brightness > 0:
 			shadow = brightness
 			highlight = 255
 		else:
 			shadow = 0
 			highlight = 255 + brightness
 		alpha_b = (highlight - shadow)/255
 		gamma_b = shadow
 		
 		buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
 	else:
 		buf = input_img.copy()
 	
 	if contrast != 0:
 		f = 131*(contrast + 127)/(127*(131-contrast))
 		alpha_c = f
 		gamma_c = 127*(1-f)
 		
 		buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
 		
 	return buf

""" -----------------------------------------------------------------------------------------------------
Build montage for generating proof
Credit: Kyle Hounslow
https://github.com/jrosebr1/imutils/blob/master/imutils/convenience.py
 -----------------------------------------------------------------------------------------------------"""
def build_montages(image_list, image_shape, montage_shape):
    image_montages = []
    # start with black canvas to draw images onto
    montage_image = np.zeros(shape=(image_shape[1] * (montage_shape[1]), image_shape[0] * montage_shape[0], 3),
                          dtype=np.uint8)
    cursor_pos = [0, 0]
    start_new_img = False
    for img in image_list:
        #if type(img).__module__ != np.__name__:
        #    raise Exception('input of type {} is not a valid numpy array'.format(type(img)))
        start_new_img = False
        img = cv2.resize(img, image_shape)
        # draw image to black canvas
        montage_image[cursor_pos[1]:cursor_pos[1] + image_shape[1], cursor_pos[0]:cursor_pos[0] + image_shape[0]] = img
        cursor_pos[0] += image_shape[0]  # increment cursor x position
        if cursor_pos[0] >= montage_shape[0] * image_shape[0]:
            cursor_pos[1] += image_shape[1]  # increment cursor y position
            cursor_pos[0] = 0
            if cursor_pos[1] >= montage_shape[1] * image_shape[1]:
                cursor_pos = [0, 0]
                image_montages.append(montage_image)
                # reset black canvas
                montage_image = np.zeros(shape=(image_shape[1] * (montage_shape[1]), image_shape[0] * montage_shape[0], 3),
                                      dtype=np.uint8)
                start_new_img = True
    if start_new_img is False:
        image_montages.append(montage_image)  # add unfinished montage
    return image_montages


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def rescale_linear(array, new_min, new_max):
    """Rescale an arrary linearly."""
    minimum, maximum = np.min(array), np.max(array)
    m = (new_max - new_min) / (maximum - minimum)
    b = new_min - m * minimum
    return m * array + b
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#  
def circ(radius, chord):
    ### has to be between -1 and 1
    inside = chord / (2*radius)
    if  -1 < inside < 1:
        centa = np.arcsin(inside)*2
        areasec = centa*(radius**2)*0.5
        areacirc = math.pi*(radius**2)
        KRN = areacirc/areasec
    else:
        print("arcsin not in bounds")
        inside = np.nan
        centa = np.nan
        areasec = np.nan
        areacirc = np.nan
        KRN = np.nan
    return inside, centa, areasec, areacirc, KRN    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


""" -----------------------------------------------------------------------------------------------------
Basic Thresholding module for background removal when k means dont work
 -----------------------------------------------------------------------------------------------------"""
def thresh(img, channel, threshold, inv, debug):

    """basic thersholding technique

    b = 
    g =
    r =
    h =
    s =
    v =
    l =
    a =
    b_chnl =

    threshold out be any number from 1< x < 254 or 'otsu'

    'inv' to invert (use for white backgrounds)

    """
    ears = img.copy()
    b,g,r = cv2.split(ears)                                         #Split into it channel constituents
    hsv = cv2.cvtColor(ears, cv2.COLOR_BGR2HSV)
    hsv[img == 0] = 0
    h,s,v = cv2.split(hsv)                                          #Split into it channel constituents
    lab = cv2.cvtColor(ears, cv2.COLOR_BGR2LAB)
    lab[img == 0] = 0
    l,a,b_chnl = cv2.split(lab)                                     #Split into it channel constituents
    
    if channel == 'b':
        channel = b
    elif channel == 'g':
        channel = g
    elif channel == 'r':
        channel = r
    elif channel == 'h':
        channel = h
    elif channel == 's':
        channel = s
    elif channel == 'v':
        channel = v
    elif channel == 'l':
        channel = l
    elif channel == 'a':
        channel = a
    elif channel == 'b_chnl':
        channel = b_chnl

    if debug is True:
        cv2.namedWindow('[DEBUG] [EARS] Channel for Thresholding', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('[DEBUG] [EARS] Channel for Thresholding', 1000, 1000)
        cv2.imshow('[DEBUG] [EARS] Channel for Thresholding', channel); cv2.waitKey(3000); cv2.destroyAllWindows()
        #plt.hist(channel.ravel(),256,[0,256]); plt.show()

    if threshold == 'otsu':
        otsu,_ = cv2.threshold(channel, 0, 255, cv2.THRESH_OTSU)
        bkgrnd = cv2.threshold(channel, int(otsu*0.75),256, cv2.THRESH_BINARY)[1]
        print("otsu found {} threshold".format(otsu))
    else:
        bkgrnd = cv2.threshold(channel, int(threshold),256, cv2.THRESH_BINARY)[1]


    if inv == "inv":
        bkgrnd=cv2.bitwise_not(bkgrnd)

    if debug is True:
        cv2.namedWindow('[DEBUG] [EARS] Channel for Thresholding', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('[DEBUG] [EARS] Channel for Thresholding', 1000, 1000)
        cv2.imshow('[DEBUG] [EARS] Channel for Thresholding', bkgrnd); cv2.waitKey(3000); cv2.destroyAllWindows()

    return bkgrnd