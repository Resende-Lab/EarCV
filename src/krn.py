#krn.py

import sys, traceback, os, re
import numpy as np
import pandas as pd
import math
import cv2
from scipy.spatial import distance as dist
from scipy import stats
from scipy.signal import savgol_filter
from scipy.signal import argrelextrema
from statsmodels.nonparametric.smoothers_lowess import lowess


from matplotlib import pyplot as plt
#import plotly.graph_objects as go
#from plotly.subplots import make_subplots


import utility

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#################################DEFINE MAIN FUNCTION#####################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def krn(img, debug):

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#################################### Slice Horizontally ##################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	ear = img.copy()		
	ear = cv2.rotate(ear, cv2.cv2.ROTATE_90_CLOCKWISE) 											#Make copy of original image
	wid = ear.copy()
	wid2 = ear.copy()

#grab lower half of the ear
	sect = utility.ranges(ear.shape[1], 6)
	see1 = sect[0].split (",")
	see2 = sect[4].split (",")
	wid[:, int(see1[1]):int(see2[0])] = 0
	wid2[wid != 0] = 0
	wid3 = wid2.copy()
	wid4 = wid3.copy()
	widk = img.copy()

#grab middle of the ear
	sect = utility.ranges(ear.shape[0], 4)
	see1 = sect[0].split (",")
	see2 = sect[3].split (",")	
	wid2[int(see1[1]):int(see2[0]), :] = 0
	wid3[wid2 != 0] = 0

#remove the rest
	gray = cv2.cvtColor(wid3, cv2.COLOR_BGR2GRAY)
	grayk = cv2.cvtColor(widk, cv2.COLOR_BGR2GRAY)
	thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
	threshk = cv2.threshold(grayk, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
	cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if len(cnts) == 2 else cnts[1]
	cntsk = cv2.findContours(threshk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cntsk = cntsk[0] if len(cntsk) == 2 else cntsk[1]
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
	cntsk = sorted(cntsk, key=cv2.contourArea, reverse=True)
	for c in cnts:
		x,y,w,h = cv2.boundingRect(c)    ###CHECJ THIS MIGHT AFFECT THE RESULTS
		wid3 = wid3[0:ear.shape[0], x:x+w, ] 		#CUT IMAGE		
		wid4 = wid4[0:ear.shape[0], x:x+w, ] 		#CUT IMAGE				
		wid3 = wid3[y:y+h, 0:ear.shape[1]]

		break
	for c in cntsk:
		x, y, w, h = cv2.boundingRect(c)  ###CHECJ THIS MIGHT AFFECT THE RESULTS
		widk = widk[y:y+h, x:x+w]  # CUT IMAGE
		break


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#################################### rotate and flatten ##################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	wid3 = cv2.rotate(wid3, cv2.ROTATE_90_COUNTERCLOCKWISE)

	img_h, img_w, _ = wid3.shape
	split_width = int(img_w)
	split_height = int(img_h/5)

	Y_points = utility.start_points(img_h, split_height, .1) ###ADD UTILITY
	qr_count = 0 

	peak_num = []
	global_diff = []
	global_sd = []
	global_peak_num = []
	global_mean_diff = []
	global_median_diff = []
	global_diff_sd = []

	krnl_proof1 = np.zeros([2, split_width], dtype = int)
	# = []numpy.ndarray

	for i in Y_points:
		split = wid3[i:i+split_height, 0:img_w]

		qr_count += 1
		#print(qr_count)
		_,g1,_ = cv2.split(split)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#################################### call peaks ##################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	#sum intestines (1D)
		x = np.sum(g1,axis=0)

	#normalize (for pltting with image)
		#x = (x - np.min(x))/np.ptp(x)
		#x = -1*x*split.shape[0]
		cols = savgol_filter(x, 81, 4)
	#find valleys
		c = argrelextrema(cols, np.less)
		c = c[0]
	#width, num, sd
		width = (len(c))
		#print(width)
		diff = np.diff(c)
		#print(diff)
		sd = np.std(diff)
		#print(sd)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#################################### accumulate ##################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
		if sd < 10:
			peak_num.append(width)
			global_diff.append(diff)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#################################### proof ##################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
		lines = c
		line_thickness = 3
		
		if qr_count == 1:
			krnl_proof1 = split.copy()
			for i in range (0, len(lines)):
				if sd < 10:
					x1, y1 = int(lines[i]), 0
					x2, y2 = int(lines[i]), krnl_proof1.shape[0]
					cv2.line(krnl_proof1, (x1, y1), (x2, y2), (219, 112, 147), thickness=line_thickness)
				else:
					x1, y1 = int(lines[i]), 0
					x2, y2 = int(lines[i]), krnl_proof1.shape[0]
					cv2.line(krnl_proof1, (x1, y1), (x2, y2), (0, 0, 225), thickness=line_thickness)					
		else:
			krn_proof = split.copy()
			for i in range (0, len(lines)):
				if sd < 10:
					x1, y1 = int(lines[i]), 0
					x2, y2 = int(lines[i]), krn_proof.shape[0]
					cv2.line(krn_proof, (x1, y1), (x2, y2), (219, 112, 147), thickness=line_thickness)
				else:
					x1, y1 = int(lines[i]), 0
					x2, y2 = int(lines[i]), krnl_proof1.shape[0]
					cv2.line(krn_proof, (x1, y1), (x2, y2), (0, 0, 225), thickness=line_thickness)					
		
			krnl_proof1 = cv2.vconcat([krnl_proof1, krn_proof])
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#################################### gloabl stats ##################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	global_peak_num = np.mean(peak_num)
	global_diff = np.concatenate(global_diff, axis=0 )
	global_mean_diff = np.mean(global_diff)
	global_diff_sd = np.std(global_diff)
	

	return global_peak_num, global_mean_diff, global_diff_sd, krnl_proof1