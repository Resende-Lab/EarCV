"""
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
##################################  FIND EARS MODULE  ###################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

This script is a set of tools used in 'main.py' to find the ears, clean the ears, remove silks, and finally orient the ears. 
This script requires that `OpenCV 2', 'numpy', and 'scipy' be installed within the Python environment you are running this script in.
This script imports the 'utility.py', module within the same folder.

"""

import numpy as np
import cv2
import argparse
import os
import utility
import sys
from statistics import stdev, mean
from scipy.spatial import distance as dist
#import args_log


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Use K means to threshold ears in lab channel
def kmeans(img):
	#log = args_log.get_logger("logger")

	lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
	vectorized = lab.reshape((-1,3))
	vectorized = np.float32(vectorized)
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	K = 2
	attempts = 3
	ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
	center = np.uint8(center)
	res = center[label.flatten()]
	img_sgmnt = res.reshape((img.shape))
	_,_,gray = cv2.split(img_sgmnt)		
	_,bkgrnd = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
	
	return bkgrnd


#~~~~~~~~~~~~~~~~~~~~~~~~~~~Filter connected components with area, aspect:ratio, and solidity	
def filter(filename, binary, min_area, max_area, aspect_ratio, solidity):
	#log = args_log.get_logger("logger")

	cnts = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cnts = cnts[0] if len(cnts) == 2 else cnts[1]
	mask = np.zeros_like(binary)
	i = 0
	for c in cnts:
		ear_area = cv2.contourArea(c)
		if max_area > ear_area > min_area:
			hulls = cv2.convexHull(c); hull_areas = cv2.contourArea(hulls)
			ear_solidity = float(ear_area)/hull_areas
			rects = cv2.minAreaRect(c)
			width_i = int(rects[1][0])
			height_i = int(rects[1][1])
			if height_i > width_i:
				rat = round(width_i/height_i, 2)
			else:
				rat = round(height_i/width_i, 2)
			if 0.1 < rat < aspect_ratio and 0.1 < ear_solidity < solidity: 
				cv2.drawContours(mask, [c], -1, (255), -1)
				i = i+1

	return mask, i


def calculate_area_cov(filtered, cov):
	cnts = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cnts = cnts[0] if len(cnts) == 2 else cnts[1]	# Calculate features of filtered ears
	areas = [cv2.contourArea(c) for c in cnts]
	if len(areas) > 1:
		cov = (stdev(areas)/mean(areas))
	else:
		cov is None
	return cov

def calculate_convexity(binary):
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Calculate convexity
	cntss = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	cntss = cntss[0] if len(cntss) == 2 else cntss[1]
	cntss = sorted(cntss, key=cv2.contourArea, reverse=False)[:len(cntss)]
	for cs in cntss:
		perimeters = cv2.arcLength(cs,True); hulls = cv2.convexHull(cs); hullperimeters = cv2.arcLength(hulls,True)
		convexity = hullperimeters/perimeters
	return convexity		

def rotate_ear(ear):
	_,_,r = cv2.split(ear)
	sect = utility.ranges(ear.shape[0], 3)
	ori_width = []
	for i in range(3):
		see = sect[i].split (",")
		wid = r.copy()
		wid2 = r.copy()
		wid[int(see[0]):int(see[1]), :] = 0
		wid2[wid != 0] = 0
		wid2 = utility.cnctfill(wid2)
		cntss = cv2.findContours(wid2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		cntss = cntss[0] if len(cntss) == 2 else cntss[1]
		cntss = sorted(cntss, key=cv2.contourArea, reverse=False)[:len(cntss)]
		for cs in cntss:
			rects = cv2.minAreaRect(cs)
			boxs = cv2.boxPoints(rects)
			boxs = np.array(boxs, dtype="int")			
			boxs = utility.order_points(boxs)
			(tls, trs, brs, bls) = boxs
			(tlblXs, tlblYs) = ((tls[0] + bls[0]) * 0.5, (tls[1] + bls[1]) * 0.5)
			(trbrXs, trbrYs) = ((trs[0] + brs[0]) * 0.5, (trs[1] + brs[1]) * 0.5)
			thmp = dist.euclidean((tlblXs, tlblYs), (trbrXs, trbrYs)) #pixel width
		ori_width.append(thmp)
	return ori_width

