"""
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
##################################  Extract ear features module  #########################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

This script is a set of tools using in 'main.py' to extract the features from each ear.
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


def extract_feats(ear, PixelsPerMetric):
	ear_proof = ear.copy()
	ymax = ear.shape[0]
	_,_,r = cv2.split(ear)											#Split into it channel constituents
	_,r = cv2.threshold(r, 0, 255, cv2.THRESH_OTSU)
	r = utility.cnctfill(r)

	cntss = cv2.findContours(r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	
	#cv2.namedWindow('[SILK CLEAN UP]', cv2.WINDOW_NORMAL)
	#cv2.resizeWindow('[SILK CLEAN UP]', 1000, 1000)
	#cv2.imshow('[SILK CLEAN UP]', r); cv2.waitKey(2000); cv2.destroyAllWindows()

	cntss = cntss[0] if len(cntss) == 2 else cntss[1]
	cntss = sorted(cntss, key=cv2.contourArea, reverse=False)[:len(cntss)]
	for cs in cntss:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
####################### Area, Convexity, Solidity, fitEllipse ############################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
		Ear_Area = cv2.contourArea(cs)
		perimeters = cv2.arcLength(cs,True); hulls = cv2.convexHull(cs); hull_areas = cv2.contourArea(hulls); hullperimeters = cv2.arcLength(hulls,True)
		Convexity = hullperimeters/perimeters
		Solidity = float(Ear_Area)/hull_areas
		Ellipse = cv2.fitEllipse(cs)
		MA = Ellipse[1][1]
		ma = Ellipse[1][0]
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
##################################### EAR BOX ############################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
		rects = cv2.minAreaRect(cs)
		boxs = cv2.boxPoints(rects)
		boxs = np.array(boxs, dtype="int")			
		boxs1 = utility.order_points(boxs)
# loop over the original points and draw them
# unpack the ordered bounding box, then compute the midpoint
		(tls, trs, brs, bls) = boxs
		(tltrXs, tltrYs) = ((tls[0] + trs[0]) * 0.5, (tls[1] + trs[1]) * 0.5)
		(blbrXs, blbrYs) = ((bls[0] + brs[0]) * 0.5, (bls[1] + brs[1]) * 0.5)
		(tlblXs, tlblYs) = ((tls[0] + bls[0]) * 0.5, (tls[1] + bls[1]) * 0.5)
		(trbrXs, trbrYs) = ((trs[0] + brs[0]) * 0.5, (trs[1] + brs[1]) * 0.5)
# compute the Euclidean distance between the midpoints
		Ear_Box_Width = dist.euclidean((tltrXs, tltrYs), (blbrXs, blbrYs)) #pixel length
		Ear_Box_Length = dist.euclidean((tlblXs, tlblYs), (trbrXs, trbrYs)) #pixel width
		Ear_Box_Area = float(Ear_Box_Length*Ear_Box_Width)

		if Ear_Box_Width > Ear_Box_Length:
			Ear_Box_Width = dist.euclidean((tlblXs, tlblYs), (trbrXs, trbrYs)) #pixel width
			Ear_Box_Length = dist.euclidean((tltrXs, tltrYs), (blbrXs, blbrYs)) #pixel length
			cv2.line(ear_proof, (int(tltrXs), int(tltrYs)), (int(blbrXs), int(blbrYs)), (165, 105, 189), 7) #length
			cv2.circle(ear_proof, (int(tltrXs), int(tltrYs)), 15, (165, 105, 189), -1) #left midpoint
			cv2.circle(ear_proof, (int(blbrXs), int(blbrYs)), 15, (165, 105, 189), -1) #right midpoint
		else:
			cv2.line(ear_proof, (int(tlblXs), int(tlblYs)), (int(trbrXs), int(trbrYs)), (165, 105, 189), 7) #length
			cv2.circle(ear_proof, (int(tlblXs), int(tlblYs)), 15, (165, 105, 189), -1) #left midpoint
			cv2.circle(ear_proof, (int(trbrXs), int(trbrYs)), 15, (165, 105, 189), -1) #right midpoint
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#################################### EXTREME POINTS ######################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
		extTops = tuple(cs[cs[:, :, 1].argmin()][0])
		extBots = tuple(cs[cs[:, :, 1].argmax()][0])
		Ear_Extreme_Length = dist.euclidean(extTops, extBots)
		cv2.circle(ear_proof, extTops, 15, (255, 255, 204), -1)
		#cv2.circle(ear_proof, extBots, 30, (156, 144, 120), -1)
		#cv2.line(ear_proof, extTops, extBots, (0, 0, 155), 10)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#################################### POLY DP Convexity ###################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
		arclen = cv2.arcLength(cs, True)
# do approx
		eps = 0.001
		epsilon = arclen * eps
		approx = cv2.approxPolyDP(cs, epsilon, True)
# draw the result
		canvas = np.zeros_like(r)
		cv2.drawContours(canvas, [approx], -1, (255), 2, cv2.LINE_AA)

	cntss = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	cntss = cntss[0] if len(cntss) == 2 else cntss[1]
	cntss = sorted(cntss, key=cv2.contourArea, reverse=False)[:len(cntss)]
	for cs in cntss:
		canvas_perimeters = cv2.arcLength(cs,True); canvas_hulls = cv2.convexHull(cs); canvas_hullperimeters = cv2.arcLength(hulls,True)
		if canvas_perimeters != 0:
			Convexity_polyDP = canvas_hullperimeters/canvas_perimeters
		else:
			Convexity_polyDP

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
########################################## Taper #########################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	taper = r.copy()
	taper[int(1-(ymax/2)): ymax, :] = 0 #FOR THE TIP
	cntss = cv2.findContours(taper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	cntss = cntss[0] if len(cntss) == 2 else cntss[1]
	cntss = sorted(cntss, key=cv2.contourArea, reverse=False)[:len(cntss)]
	for cs in cntss:
		Taper_Area = cv2.contourArea(cs)
		Taper_perimeters = cv2.arcLength(cs,True); Taper_hulls = cv2.convexHull(cs); Taper_hull_areas = cv2.contourArea(hulls); Taper_hullperimeters = cv2.arcLength(hulls,True)

		if Taper_perimeters != 0:
			Taper_Convexity = Taper_hullperimeters/Taper_perimeters
		else:
			Taper_Convexity = None
		
		if Taper_hull_areas != 0:
			Taper_Solidity = float(Taper_Area)/Taper_hull_areas
		else:
			Taper_Solidity = None

	taper_canvas = canvas.copy()
	taper_canvas = utility.cnctfill(taper_canvas)
	taper_canvas=cv2.bitwise_not(taper_canvas)
	taper_canvas[int(1-(ymax/2)): ymax, :] = 0 #FOR THE TIP
	cntss = cv2.findContours(taper_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	cntss = cntss[0] if len(cntss) == 2 else cntss[1]
	cntss = sorted(cntss, key=cv2.contourArea, reverse=False)[:len(cntss)]
	for cs in cntss:
		canvas_perimeters = cv2.arcLength(cs,True); canvas_hulls = cv2.convexHull(cs); canvas_hullperimeters = cv2.arcLength(canvas_hulls,True)
		if canvas_perimeters != 0:
			Taper_Convexity_polyDP = canvas_hullperimeters/canvas_perimeters
		else:
			Taper_Convexity_polyDP = None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
######################################### WIDTHS #########################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	Cents = []
	Widths = []
	wid_proof = ear.copy()	
	sect = utility.ranges(ear.shape[0], 20)

	for i in range(20):
		see = sect[i].split (",")
		wid = r.copy()
		wid2 = r.copy()
		wid[int(see[0]):int(see[1]), :] = 0
		wid2[wid != 0] = 0
		wid2 = utility.cnctfill(wid2)
		if cv2.countNonZero(wid2) != 0:
			cntss = cv2.findContours(wid2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
			cntss = cntss[0] if len(cntss) == 2 else cntss[1]
			cntss = sorted(cntss, key=cv2.contourArea, reverse=False)[:len(cntss)]
			M1 = cv2.moments(wid2)
			if M1["m00"] != 0:
				cX = int(M1["m10"] / M1["m00"])
				Cents.append(cX)
				cY = int(M1["m01"] / M1["m00"])
				for cs in cntss:
					rects = cv2.minAreaRect(cs)
					boxs = cv2.boxPoints(rects)
					boxs = np.array(boxs, dtype="int")			
					boxs = utility.order_points(boxs)
					(tls, trs, brs, bls) = boxs
					(tlblXs, tlblYs) = ((tls[0] + bls[0]) * 0.5, (tls[1] + bls[1]) * 0.5)
					(trbrXs, trbrYs) = ((trs[0] + brs[0]) * 0.5, (trs[1] + brs[1]) * 0.5)
					Ear_Width_B = dist.euclidean((tlblXs, tlblYs), (trbrXs, trbrYs)) #pixel width
					Widths.append(Ear_Width_B)
					cv2.line(wid_proof, (int(tlblXs), int(tlblYs)), (int(trbrXs), int(trbrYs)), (33, 43, 156), 7) #width
				cv2.circle(wid_proof, (cX, cY), 20, (176, 201, 72), -1)

	Widths_Sdev = stdev(Widths)		
	Cents_Sdev = stdev(Cents)
	Taper =  stdev(Widths[0:10])

	####CONVERT PIXELS EPR MERTIC SHIT
	if PixelsPerMetric is not None:
		Ear_Area = Ear_Area / (PixelsPerMetric*PixelsPerMetric)
		Ear_Box_Area = Ear_Box_Area / (PixelsPerMetric*PixelsPerMetric)
		Ear_Box_Length = Ear_Box_Length / (PixelsPerMetric)
		Ear_Box_Width = Ear_Box_Width / (PixelsPerMetric)
		Ear_Extreme_Length = Ear_Extreme_Length / (PixelsPerMetric)
		perimeters = perimeters	/ (PixelsPerMetric)		
		newWidths = [x / PixelsPerMetric for x in Widths]
		max_Width = max(Widths)
		max_Width = max_Width / (PixelsPerMetric)
		MA = MA / (PixelsPerMetric)
		ma = ma / (PixelsPerMetric)	
	else:
		newWidths = Widths
		max_Width = max(Widths)

	return	Ear_Area, Ear_Box_Area, Ear_Box_Length, Ear_Extreme_Length, Ear_Box_Width, newWidths, max_Width, MA, ma, perimeters, Convexity, Solidity, Convexity_polyDP, Taper, Taper_Convexity, Taper_Solidity, Taper_Convexity_polyDP, Widths_Sdev, Cents_Sdev, ear_proof, canvas, wid_proof


def extract_moments(ear):
	_,_,r = cv2.split(ear)											#Split into it channel constituents
	_,r = cv2.threshold(r, 0, 255, cv2.THRESH_OTSU)
	r = utility.cnctfill(r)
	cntss = cv2.findContours(r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	cntss = cntss[0] if len(cntss) == 2 else cntss[1]
	cntss = sorted(cntss, key=cv2.contourArea, reverse=False)[:len(cntss)]

	for cs in cntss:
		moments = cv2.moments(cs)
		#print M

	return	moments


def krnl_feats(ear, tip, bottom, PixelsPerMetric):			
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	################################## KERNEL FEATS ##########################################
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

	krnl_proof = ear.copy()
	_,_,r = cv2.split(ear)											#Split into it channel constituents
	_,r = cv2.threshold(r, 0, 255, cv2.THRESH_OTSU)
	r = utility.cnctfill(r)

	Ear_area = cv2.countNonZero(r)
	
	cob = tip + bottom
	uncob = ear.copy()
	krnl = ear.copy()
	uncob[cob == 255] = 0
	_,_,uncob = cv2.split(uncob)
	_,uncob = cv2.threshold(uncob, 0, 255, cv2.THRESH_OTSU)
	uncob = utility.cnctfill(uncob)
	krnl[uncob == 0] = 0

	pixels = np.float32(krnl[uncob !=  0].reshape(-1, 3))
			
	Blue, Green, Red, Hue, Sat, Vol, Light, A_chnnl, B_chnnl = dominant_cols(krnl, pixels)
	#frame_fr = np.zeros_like(krnl)
	#frame_fr[uncob > 0] = [Blue, Red, Green]

	Tip_Area = cv2.countNonZero(tip)
	Bottom_Area = cv2.countNonZero(bottom)
	Krnl_Area = cv2.countNonZero(uncob)

	Tip_Fill = (Ear_area-Tip_Area)/Ear_area
	Bottom_Fill = (Ear_area-Bottom_Area)/Ear_area
	Krnl_Fill = (Ear_area-Krnl_Area)/Ear_area

	cntss, _ = cv2.findContours(uncob, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	for cs in cntss:
		krnl_perimeters = cv2.arcLength(cs,True); krnl_hulls = cv2.convexHull(cs); krnl_hullperimeters = cv2.arcLength(krnl_hulls,True)
		if krnl_perimeters != 0:
			Krnl_Convexity = krnl_hullperimeters/krnl_perimeters
		
		krnl_proof =cv2.drawContours(krnl_proof, [cs], -1, ([int(Blue), int(Green), int(Red)]), -1)	
		rects = cv2.minAreaRect(cs)
		boxs = cv2.boxPoints(rects)
		boxs = np.array(boxs, dtype="int")			
		boxs1 = utility.order_points(boxs)
		# loop over the original points and draw them
		# unpack the ordered bounding box, then compute the midpoint
		(tls, trs, brs, bls) = boxs1
		(tltrXs, tltrYs) = ((tls[0] + trs[0]) * 0.5, (tls[1] + trs[1]) * 0.5)
		(blbrXs, blbrYs) = ((bls[0] + brs[0]) * 0.5, (bls[1] + brs[1]) * 0.5)
		# compute the Euclidean distance between the midpoints
		Kernel_Length = dist.euclidean((tltrXs, tltrYs), (blbrXs, blbrYs)) #pixel length

	
	if PixelsPerMetric is not None:
		Tip_Area = Tip_Area / (PixelsPerMetric*PixelsPerMetric)
		Bottom_Area = (PixelsPerMetric*PixelsPerMetric)
		Krnl_Area = (PixelsPerMetric*PixelsPerMetric)
		Kernel_Length = Kernel_Length / (PixelsPerMetric)
	
	return Tip_Area, Bottom_Area, Krnl_Area, Kernel_Length, Krnl_Convexity, Tip_Fill, Bottom_Fill, Krnl_Fill, krnl_proof, cob, uncob, Blue, Red, Green, Hue, Sat, Vol, Light, A_chnnl, B_chnnl


def dominant_cols(krnl, pixels):
#~~~~~~~~~~~~~~~~~~~~~~~~~Dominant Color
	n_colors = 2
	_,_,r = cv2.split(krnl)											#Split into it channel constituents
	hsv = cv2.cvtColor(krnl, cv2.COLOR_BGR2HSV)
	hsv[krnl == 0] = 0
	lab = cv2.cvtColor(krnl, cv2.COLOR_BGR2LAB)
	lab[krnl == 0] = 0



	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
	flags = cv2.KMEANS_RANDOM_CENTERS
	_, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
	_, counts = np.unique(labels, return_counts=True)
	dominant = palette[np.argmax(counts)]
	Red = dominant[0]
	Green = dominant[1]
	Blue = dominant[2]
#~~~~~~~~~~~~~~~~~~~~~~~~~Dominant HSV Color
	hsv = cv2.cvtColor(krnl, cv2.COLOR_BGR2HSV)						#Convert into HSV color Space
	pixels = np.float32(hsv[r != 0].reshape(-1, 3))
	n_colors = 2
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
	flags = cv2.KMEANS_RANDOM_CENTERS
	_, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
	_, counts = np.unique(labels, return_counts=True)
	hsv_dominant = palette[np.argmax(counts)]
	Hue = hsv_dominant[0]
	Sat = hsv_dominant[1]
	Vol = hsv_dominant[2]

#~~~~~~~~~~~~~~~~~~~~~~~~~Dominant LAB Color
	lab = cv2.cvtColor(krnl, cv2.COLOR_BGR2LAB)						#Convert into HSV color Space
	pixels = np.float32(lab[r != 0].reshape(-1, 3))
	n_colors = 2
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
	flags = cv2.KMEANS_RANDOM_CENTERS
	_, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
	_, counts = np.unique(labels, return_counts=True)
	lab_dominant = palette[np.argmax(counts)]
	Light = lab_dominant[0]
	A_chnnl = lab_dominant[1]
	B_chnnl = lab_dominant[2]

	return Red, Green, Blue, Hue, Sat, Vol, Light, A_chnnl, B_chnnl