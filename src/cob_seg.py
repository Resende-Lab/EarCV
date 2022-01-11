#tip_seg.py
import numpy as np
import cv2
import argparse
import os
import utility
import sys
from statistics import stdev, mean
from scipy.spatial import distance as dist
#import args_log

#np.argsort(-stats[:,-1])
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
##########################################TIP#############################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#	
def kmeans(chnnl):
	
	chnnl = cv2.cvtColor(chnnl,cv2.COLOR_GRAY2RGB)	
	#k means implementation
	vectorized = chnnl.reshape((-1,3))
	vectorized = np.float32(vectorized)
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	K = 2
	attempts = 5
	ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
	center = np.uint8(center)
	res = center[label.flatten()]
	cob= res.reshape((chnnl.shape))

	cob = cv2.cvtColor(cob,cv2.COLOR_RGB2GRAY)
	_,cob = cv2.threshold(cob, 0, 255, cv2.THRESH_OTSU)

	return cob

def otsu(chnnl):
	
	otsu,cob = cv2.threshold(chnnl, 0, 255, cv2.THRESH_OTSU)	

	return cob, otsu

def manual(chnnl, threshold):

	cob = cv2.threshold(chnnl, int(threshold),256, cv2.THRESH_BINARY)[1]
	
	return cob


def top_modifier(ear, tip, tip_percent, dialate, debug):
	ymax = ear.shape[0]
	_,_,red = cv2.split(ear)											#Split into it channel constituents
	_,red = cv2.threshold(red, 0, 255, cv2.THRESH_OTSU)
	tip[red == 0] = 0
	
	tip[int((ymax*(int(tip_percent)/100))):ymax, :] = 0 #FOR THE TIP

	if debug is True:
		cv2.namedWindow('[DEBUG][EAR] Tip Percent', cv2.WINDOW_NORMAL)
		cv2.resizeWindow('[DEBUG][EAR] Tip Percent', 1000, 1000)
		cv2.imshow('[DEBUG][EAR] Tip Percent', tip); cv2.waitKey(3000); cv2.destroyAllWindows()

	#tip = cv2.morphologyEx(tip, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (int(dialate),int(dialate))), iterations=int(2))
	tip = cv2.dilate(tip, cv2.getStructuringElement(cv2.MORPH_RECT, (int(dialate),int(dialate))), iterations=int(2))
	
	if debug is True:
		cv2.namedWindow('[DEBUG][EAR] Tip Dialate', cv2.WINDOW_NORMAL)
		cv2.resizeWindow('[DEBUG][EAR] Tip Dialate', 1000, 1000)
		cv2.imshow('[DEBUG][EAR] Tip Dialate', tip); cv2.waitKey(3000); cv2.destroyAllWindows()
	
	if cv2.countNonZero(tip) != 0:
		tip2 = red.copy()
		red[tip == 255] = 0
		red = utility.cnctfill(red)
		tip2[red == 255] = 0
		tip = tip2.copy()

		if debug is True:
			cv2.namedWindow('[DEBUG][EAR] Tip Processed (after invert)', cv2.WINDOW_NORMAL)
			cv2.resizeWindow('[DEBUG][EAR] Tip Processed (after invert)', 1000, 1000)
			cv2.imshow('[DEBUG][EAR] Tip Processed (after invert)', tip2); cv2.waitKey(3000); cv2.destroyAllWindows()

	return tip

def bottom_modifier(ear, bottom, bottom_percent, dialate, debug):
	ymax = ear.shape[0]
	_,_,red = cv2.split(ear)											#Split into it channel constituents
	_,red = cv2.threshold(red, 0, 255, cv2.THRESH_OTSU)
	bottom[red == 0] = 0

	bottom[0:int((ymax*(int(bottom_percent)/100))), :] = 0 #FOR THE bottom

	if debug is True:
		cv2.namedWindow('[DEBUG][EAR] Bottom Percent', cv2.WINDOW_NORMAL)
		cv2.resizeWindow('[DEBUG][EAR] Bottom Percent', 1000, 1000)
		cv2.imshow('[DEBUG][EAR] Bottom Percent', bottom); cv2.waitKey(3000); cv2.destroyAllWindows()
	
	bottom = cv2.dilate(bottom, cv2.getStructuringElement(cv2.MORPH_RECT, (int(dialate),int(dialate))), iterations=int(2))

	if debug is True:
		cv2.namedWindow('[DEBUG][EAR] Bottom Dialate', cv2.WINDOW_NORMAL)
		cv2.resizeWindow('[DEBUG][EAR] Bottom Dialate', 1000, 1000)
		cv2.imshow('[DEBUG][EAR] Bottom Dialate', bottom); cv2.waitKey(3000); cv2.destroyAllWindows()

	
	if cv2.countNonZero(bottom) != 0:
		bottom2 = red.copy()
		red[bottom == 255] = 0
		red = utility.cnctfill(red)
		bottom2[red == 255] = 0
		bottom = bottom2.copy()


		if debug is True:
			cv2.namedWindow('[DEBUG][EAR] Bottom Processed (after invert)', cv2.WINDOW_NORMAL)
			cv2.resizeWindow('[DEBUG][EAR] Bottom Processed (after invert)', 1000, 1000)
			cv2.imshow('[DEBUG][EAR] Bottom Processed (after invert)', bottom); cv2.waitKey(3000); cv2.destroyAllWindows()


	return bottom