# -*- coding: utf-8 -*-
"""--------------------------------------------------------------------------------------------------

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
###################### Computer Vision for Maize Ear Analysis  ###########################
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
###################### By: Juan M. Gonzalez, University of Florida  ######################
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This tool allows the user to rapidly extract features from images containing maize ears against a uniform background.
This tool requires that `OpenCV 2', 'numpy', 'scipy', 'pyzbar'(optional), and 'plantcv(optinal)' be installed within the Python
environment you are running this script in.

This file imports modules within the same folder:

	* args_log.py - Cpntains the arguments and log settings for the main script.
    * qr.py - Scans image for QR code and returns found information.
    * ColorCorrection.py - Performs color correction on images using a color checker.
    * ppm.py - Calculates the pixels per metric using a solid color square in the input image of known dimensions.
    * find_ears.py - Segments ears in the input image.
    * features.py - Measures basic ear morphological features and kernel features.
    * cob_chank_segmentation.py - Segments kernel from cob and shank.
    * krn.py - Counts kernel peaks and estimates median kernel width
	* utilities.py - Helper functions needed thorughout the analysis.

--------------------------------------------------------------------------------------------------"""

import sys
import traceback
import os
import re
import string
import csv
import imghdr
import numpy as np
import cv2
from statistics import stdev, mean
from scipy.spatial import distance as dist
from plantcv import plantcv as pcv

import utility
import args_log
import qr
import clr
import ppm
import find_ears
import features
import cob_seg
import krn



#import entropy

#from earcv import __version__

__author__ = "Juan M. Gonzalez"
__copyright__ = "Juan M. Gonzalez"
__license__ = "mit"

def main():

	"""Full pipeline for automated maize ear phenotyping.

    This tool allows the user to rapidly extract ear features from images containing maize ears against a uniform background.

    Parameters
    ----------
    **kwargs : iterable
        [-h] -i IMAGE [-o OUTDIR] [-ns] [-np] [-D] [-qr] [-r]
        [-qr_scan [Window size of x pixels by x pixels]
        [Amount of overlap 0 < x < 1]] [-clr COLOR_CHECKER]
        [-ppm [reference length] [in/cm]]
        [-filter [Min area as % of total image area]
        [Max Area as % of total image area] [Max Aspect Ratio]
        [Max Solidity]] [-clnup [Max area COV] [Max iterations]]
        [-slk [Min delta convexity change] [Max iterations]]
        [-t [Tip percent] [Contrast] [Threshold] [dialate]]
        [-b [Bottom percent] [Contrast] [Threshold] [dialate]]
       
        Required:

        -i, --image         Path to input image file, required. Accepted formats: 'tiff', 'jpeg', 'bmp', 'png'
        
        Optional:

        -o, --OUTDIR        Provide directory to saves proofs, logfile, and output CSVs. Default: Will save in current directory if not provided.
        -ns, --no_save      Default saves proofs and output CSVs. Raise flag to stop saving.
        -np, --no_proof     Default prints proofs on screen. Raise flag to stop printing proofs.
        -D, --debug         Raise flag to print intermediate images throughout analysis. Useful for troubleshooting.
        
        -qr, --qrcode                          Raise flag to scan entire image for QR code.
        -r, --rename                           Default renames images with found QRcode. Raise flag to stop renaming images with found QR code.
        -qr_scan, --qr_window_size_overlap     Advanced QR code scanning by breaking the image into subsections. [Window size of x pixels by x pixels] [Amount of overlap (0 < x < 1)] Provide the pixel size of square window to scan through image for QR code and the amount of overlap between sections (0 < x < 1).
        -clr, --color_checker COLOR_CHECKER    Path to input image file with reference color checker.
        -ppm, --pixelspermetric                [Refference length] [in/cm] Calculate pixels per metric using either a color checker or the largest uniform color square. Provide reference length in 'in' or 'cm'.
        

        -filter, --ear_filter       [Min area as % of total image area] [Max Area as % of total image area] [Max Aspect Ratio] [Max Solidity] Ear segmentation filter, filters each segmented object based on area, aspect ratio, and solidity. Default: Min Area--1 percent, Max Area--x percent, Max Aspect Ratio: x < 0.6, Max Solidity: 0.98. Flag with three arguments to customize ear filter.
        -clnup, --ear_cleanup       [Max area COV] [Max iterations] Ear clean-up module. Default: Max Area Coefficient of Variation threshold: 0.2, Max number of iterations: 10. Flag with two arguments to customize clean up module.
        -slk, --silk_cleanup        [Min delta convexity change] [Max iterations] Silk decontamination module. Default: Min change in covexity: 0.04, Max number of iterations: 10. Flag with two arguments to customize silk clean up module.
        

        -t, --tip [Tip percent] [Contrast] [Threshold] [dialate] Tip segmentation module. Tip percent, Contrast, Threshold, dialate. Flag with four arguments to customize tip segmentation module. Use module defaults by providing '0' for all arguments.
        -b, --bottom [Bottom percent] [Contrast] [Threshold] [dialate] Bottom segmentation module. Bottom percent, Contrast, Threshold, dialate. Flag with four arguments to customize tip segmentation module. Use module defaults module by providing '0' for all arguments.
    
    Returns
    -------
    qrcode.csv
        .csv file with the file name and the corresponding information found in QR code.
    color_check.csv
        .csv file with color correction preformance metrics based on root mean squared differences in color.
    features.csv
        .csv file with the ear features as columns and ears as rows.
    proof
        proooooooooofs

   
    Other Parameters
    ----------------
    **kwargs : iterable
        -h, --help          Show help message and exit

    Raises
    ------
    Exception
        Invalid file type. Only accepts: 'tiff', 'jpeg', 'bmp', 'png'.

    Notes
    -----
    Notes about the implementation algorithm (if needed).

    References
    ----------
    .. [1] O. McNoleg, "The integration of GIS, remote sensing,
       expert systems and adaptive co-kriging for environmental habitat
       modelling of the Highland Haggis using object-oriented, fuzzy-logic
       and neural-network techniques," Computers & Geosciences, vol. 22,
       pp. 585-588, 1996.

    Examples
    --------

    python etc...

    """


	args = args_log.options()											# Get options
	log = args_log.get_logger("logger")									# Create logger
	log.info(args)														# Print expanded arguments
	
	if args.outdir is not None:											# If out dir is provided, else use current dir
		out = args.outdir
	else:
		out = "./"
	
	fullpath, root, filename, ext = utility.img_parse(args.image)		# Parse provided path
	
	if imghdr.what(fullpath) is None:									# Is the image path valid?
		log.warning("[ERROR]--{}--Invalid image file provided".format(fullpath)) # Log
		raise Exception										

	log.info("[START]--{}--Starting analysis pipeline...".format(filename)) # Log
	img=cv2.imread(fullpath)											# Read img in
	img_h = img.shape[0]
	img_w = img.shape[1] 
	if img.shape[0] > img.shape[1]:										# Rotate the image in case it is saved vertically  
		img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
		img_h = img.shape[0]
		img_w = img.shape[1]

	log.info("[START]--{}--Image is {} by {} pixels".format(filename, img.shape[0], img.shape[1])) # Log
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	##################################  QR code module  ######################################
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

	if args.qrcode is True or args.qr_window_size_overlap is not None:	# Turn module on
		QRcodeType = QRcodeData = QRcodeRect = qr_window_size = overlap = qr_proof = removesticker = None 	# Empty variables
		log.info("[QR]--{}--Starting QR code extraction module...".format(filename))		# Log

		if args.qr_window_size_overlap is not None:								# Turn flag on to break image into subsections and flag each sub section and read in each variable
			qr_window_size =  args.qr_window_size_overlap[0]
			overlap = args.qr_window_size_overlap[1]
			log.info("[QR]--{}--Dividing image into windows of {} by {} pixels with overlap {}".format(filename, qr_window_size, qr_window_size, overlap))
			
		QRcodeType, QRcodeData, QRcodeRect, qr_count, qr_proof, removesticker = qr.qr_scan(img, qr_window_size, overlap, args.debug)	# Run the qr.py module

		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~QRCODE output~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~		
		if QRcodeData is None:										
			log.warning("[QR]--{}--Error: QR code not found".format(filename))	# Print error if no QRcode found
			qr_proof = mask = np.zeros_like(img)
			cv2.putText(qr_proof, "QR code not found", (int(500), int(500)), cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 0, 255), 15)	
		else:
			log.info("[QR]--{}--Found {}: {} on the {}th iteration".format(filename, QRcodeType, QRcodeData, qr_count))	# Log
			cv2.putText(qr_proof, "Found: {}".format(QRcodeData), (int(qr_proof.shape[0]/10), int(qr_proof.shape[1]/2)), cv2.FONT_HERSHEY_SIMPLEX,7, (222, 201, 59), 12)	# Print Text into proof
			(x, y, w, h) = QRcodeRect											# Pull coordinates for barcode location
			cv2.rectangle(qr_proof, (x, y), (x + w, y + h), (0, 0, 255), 20)	#Draw a box around the found QR code
			
			#remove sticker form image		
			removesticker = cv2.dilate(removesticker,np.ones((5,5),np.uint8),iterations = 5)
			img[removesticker != 0] = 0

			if args.debug is True:											# Print proof with QR code
				cv2.namedWindow('[DEBUG] [QR] QRcode Proof', cv2.WINDOW_NORMAL)
				cv2.resizeWindow('[DEBUG] [QR] QRcode Proof', 1000, 1000)
				cv2.imshow('[DEBUG] [QR] QRcode Proof', qr_proof); cv2.waitKey(3000); cv2.destroyAllWindows()
				
			if args.rename is True:												# Rename image with QR code
				os.rename(args.image, root + QRcodeData + ext)
				filename = QRcodeData
				log.info("[QR]--{}--Renamed with QRCODE info: {}".format(filename, filename, QRcodeData))

			if args.no_save is False:								
				csvname = out + 'QRcodes' +'.csv'								# Create CSV and store barcode info
				file_exists = os.path.isfile(csvname)
				with open (csvname, 'a') as csvfile:
					headers = ['Filename', 'QR Code']
					writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n',fieldnames=headers)
					if not file_exists:
						writer.writeheader()
					writer.writerow({'Filename': filename, 'QR Code': QRcodeData})
				log.info("[QR]--{}--Saved filename and QRcode info to: {}QRcodes.csv".format(filename, out))
			
	else:
		log.info("[QR]--{}--QR module turned off".format(filename))
		QRcodeType = QRcodeData = QRcodeRect = qr_window_size = overlap = qr_proof = removesticker = None
		qr_proof = mask = np.zeros_like(img)
		cv2.putText(qr_proof, "QR module off", (int(500), int(500)), cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 0, 255), 15)	


	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	##############################  PIXELS PER METRIC MODULE  ################################
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	if args.pixelspermetric is not None:
		PixelsPerMetric = None
		Units = args.pixelspermetric[1]
		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~PPM module Output
		log.info("[PPM]--{}--Looking for solid color square to calculate pixels per metric...".format(filename))
		PixelsPerMetric, ppm_proof = ppm.ppm_square(img, float(args.pixelspermetric[0]))	#Run the pixels per metric module without color checker
		
		if PixelsPerMetric is not None:
			log.info("[PPM]--{}--Found {} pixels per {}".format(filename, PixelsPerMetric, Units))	
			
			if args.debug is True:
				cv2.namedWindow('[DEBUG] [PPM] Pixels Per Metric: FOUND', cv2.WINDOW_NORMAL)
				cv2.resizeWindow('DEBUG] [PPM] Pixels Per Metric: FOUND', 1000, 1000)
				cv2.imshow('[DEBUG] [PPM] Pixels Per Metric: FOUND', ppm_proof); cv2.waitKey(3000); cv2.destroyAllWindows()

		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ pixels per metric csv			
			if args.no_save is False:		
				csvname = out + 'pixelspermetric' +'.csv'
				file_exists = os.path.isfile(csvname)
				with open (csvname, 'a') as csvfile:
					headers = ['Filename', 'Pixels Per Metric']
					writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n',fieldnames=headers)
					if not file_exists:
						writer.writeheader()  # file doesn't exist yet, write a header	
					writer.writerow({'Filename': filename, 'Pixels Per Metric': PixelsPerMetric})
				log.info("[PPM]--{}--Saved pixels per {} to: {}pixelspermetric.csv".format(filename, Units, out))	
				if Units == 'cm':
					Units = 'cm/cm^2'
				elif Units == 'in':
					Units = 'in/in^2'		
		else:
			log.warning("[PPM]--{}--No size reference found for pixel per metric calculation".format(filename))
			PixelsPerMetric = None
			Units = None
			ppm_proof = np.zeros_like(img)
			cv2.putText(ppm_proof, "PPM module off", (int(500), int(500)), cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 0, 255), 15)

	else:
		log.info("[PPM]--{}--Pixels per Metric module turned off".format(filename))
		PixelsPerMetric = None
		Units = None
		ppm_proof = np.zeros_like(img)
		cv2.putText(ppm_proof, "PPM module off", (int(500), int(500)), cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 0, 255), 15)	

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	##############################  Color correction module  #################################
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	
	tar_check = None

	if args.color_checker != "None" and args.color_checker != "":
		
		reff_fullpath, reff_root, reff_filename, reff_ext = utility.img_parse(args.color_checker)		# Parse provided path for reference color checker image

		if imghdr.what(reff_fullpath) is None:
			log.warning("[ERROR]--{}--Invalid reference image file provided".format(reff_fullpath)) 		# RUN A TEST HERE IF IMAGE IS REAL
			raise Exception	

		log.info("[COLOR]--{}--Starting color correction module with provided color checker reference...".format(filename)) # Log
		
		reff=cv2.imread(reff_fullpath)
		color_proof, tar_chk, corrected, avg_tar_error, avg_trans_error, csv_field = clr.color_correct(filename, img, reff, args.debug)	#Run the color correction module

	elif args.color_checker != "None":
		reff = None
		log.info("[COLOR]--{}--No reference color checker provided. Starting color correction module using hardcoded values...".format(filename)) # Log
		color_proof, tar_chk, corrected, avg_tar_error, avg_trans_error, csv_field = clr.color_correct(filename, img, reff, args.debug)	#Run the color correction module

	else:
		tar_chk = None
		log.info("[COLOR]--{}--Color correction module turned off".format(filename))
		color_proof = mask = np.zeros_like(img)
		color_proof = mask = np.zeros_like(img)
		cv2.putText(color_proof, "Color correction module off", (int(500), int(500)), cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 0, 255), 15)	
		 
	if tar_check is not None:
		log.info("[COLOR]--{}--Before correction - {} After correction - {}".format(filename, avg_tar_error, avg_trans_error)) # Log

		img[tar_chk != 0] = 0															# Mask out found color checker

		csvname = out + 'color_correction' +'.csv'										# Save results into csv
		file_exists = os.path.isfile(csvname)
		with open (csvname, 'a') as csvfile:
			headers = ['Filename', 'Overall improvement', 'Square1', 'Square1', 'Square3', 'Square4', 'Square5', 'Square6',
               'Square7', 'Square8', 'Square9', 'Square10', 'Square11', 'Square12', 'Square13',
               'Square14', 'Square15', 'Square16', 'Square17', 'Square18', 'Square19', 'Square20',
               'Square21', 'Square22', 'Square23', 'Square24']
			writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n',fieldnames=headers)
			if not file_exists:
				writer.writeheader()  													# file doesn't exist yet, write a header	

			writer.writerow({headers[i]: csv_field[i] for i in range(26)})
	
		log.info("[COLOR]--{}--Saved features to: {}color_correction.csv".format(filename, out)) # Log
		img = corrected

		if args.debug is True:
			cv2.namedWindow('[DEBUG] [CLR] Color correction: DONE', cv2.WINDOW_NORMAL)
			cv2.resizeWindow('[DEBUG] [CLR] Color correction: DONE', 1000, 1000)
			cv2.imshow('[DEBUG] [CLR] Color correction: DONE', img); cv2.waitKey(3000); cv2.destroyAllWindows()

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	##################################  Find ears module  ####################################
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	log.info("[EARS]--{}--Looking for ears...".format(filename))
	img_area = img_w*img_h
	ears_proof = img.copy()
	bkgrnd = img.copy()

	if args.threshold is None:
		bkgrnd = find_ears.kmeans(img)									# Use kmeans to segment everything from the background
	else:
		channel = args.threshold[0]
		threshold = args.threshold[1]
		inv = args.threshold[2]
		bkgrnd = utility.thresh(img,channel,threshold, inv, args.debug)									# Manually threshold the thing
	
	if args.ear_size is not None:
		log.info("[EARS]--{}--Segmenting ears with custom size filter: Min Area: {}%, Max Area: {}%".format(filename, args.ear_size[0], args.ear_size[1]))
		min_area = img_area*((args.ear_size[0])/100)
		max_area = img_area*((args.ear_size[1])/100)
	else:
		log.info("[EARS]--{}--Segmenting ears with default size filter: Min Area: 1.5%, Max Area: 15%".format(filename))
		min_area = img_area*0.0150
		max_area = img_area*0.150

	if args.ear_filter is not None:
		log.info("[EARS]--{}--Filtering ears with custom settings: {} < Aspect Ratio < {}, {} Solidity < {}".format(filename, args.ear_filter[0], args.ear_filter[1], args.ear_filter[2], args.ear_filter[3]))
		min_aspect_ratio = args.ear_filter[0]
		max_aspect_ratio = args.ear_filter[1]
		min_solidity = args.ear_filter[2]
		max_solidity = args.ear_filter[3]
	else:
		min_aspect_ratio = 0.19
		max_aspect_ratio = 0.6
		min_solidity = 0.74
		max_solidity = 0.983
		log.info("[EARS]--{}--Filtering ears with default settings: 0.19 < Aspect Ratio < 0.6, 0.74 < Solidity < 0.983".format(filename))
	
	filtered, ear_number = find_ears.filter(filename, bkgrnd, min_area, max_area, min_aspect_ratio, max_aspect_ratio, min_solidity, max_solidity)		# Run the filter module

	log.info("[EARS]--{}--Found {} Ear(s) before clean up".format(filename, ear_number))

	if ear_number == 0:									# Is the image path valid?
		log.warning("[ERROR]--{}--No ears found...aborting".format(fullpath)) # Log
		raise Exception	

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	##################################  Clean-Up Module  ####################################
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	if args.ear_cleanup != "None" and args.ear_cleanup != "":
		log.info("[CLNUP]--{}--Ear clean-up module with custom settings".format(filename))

		cov = None
		cov = find_ears.calculate_area_cov(filtered, cov)																#Calculate area coeficient of variance

		if cov is None:
			log.info("[CLNUP]--{}--Cannot calculate Coefficent of Variance on single ear".format(filename))
		else:
			log.info("[CLNUP]--{}--Area Coefficent of Variance: {}".format(filename, cov))
		
		max_cov = args.ear_cleanup[0]	
		max_iterations = args.ear_cleanup[1]
		i = 1
		while cov > max_cov  and i <= max_iterations:
			log.info("[CLNUP]--{}--Ear clean-up module: Iterate up to {} times or until area COV < {}. Current COV: {} and iteration {}".format(filename, max_iterations, max_cov, round(cov, 3), i))
			mask = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (i,i)), iterations=i)
			filtered, ear_number = find_ears.filter(filename, mask, min_area, max_area, aspect_ratio, solidity)		# Run the filter module
			cov = find_ears.calculate_area_cov(filtered)																# Calculate area coeficient of variance			
			i = i+1
		log.info("[CLNUP]--{}--Ear clean-up module finished. Final Area COV--{}".format(filename, cov))

	elif args.ear_cleanup != "None":
		cov_default_tresh = 0.30
		log.info("[CLNUP]--{}--Ear clean-up module turned on with default threshold of {}".format(filename, cov_default_tresh))

		cov = None
		cov = find_ears.calculate_area_cov(filtered, cov)																#Calculate area coeficient of variance

		if cov is None:
			log.info("[CLNUP]--{}--Cannot calculate Coefficent of Variance on single ear".format(filename))
		else:
			log.info("[CLNUP]--{}--Area Coefficent of Variance: {}".format(filename, cov))

			if cov > cov_default_tresh:
				log.warning("[CLNUP]--{}--COV {} is above default threshold {} has triggered default ear clean-up module".format(filename, cov, cov_default_tresh))
				max_cov = 0.30
				max_iterations = 10
				i = 1
				while cov > max_cov  and i <= max_iterations:
					log.info("[CLNUP]--{}--Ear clean-up module: Iterate up to {} times or until area COV < {}. Current COV: {} and iteration {}".format(filename, max_iterations, max_cov, round(cov, 3), i))
					mask = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (i,i)), iterations=i)
					filtered, ear_number = find_ears.filter(filename, mask, min_area, max_area, aspect_ratio, solidity)		# Run the filter module
					cov = find_ears.calculate_area_cov(filtered, cov)																# Calculate area coeficient of variance			
					i = i+1
				log.info("[CLNUP]--{}--Ear clean-up module finished. Final Area COV--{}".format(filename, cov))
			else:
				log.info("[CLNUP]--{}--Area COV under threshold. Ear clean-up module turned off.".format(filename))
	else:
		log.info("[CLNUP]--{}--Ear clean-up module turned off.".format(filename))


	if ear_number == 0:									# Is the image path valid?
		log.warning("[ERROR]--{}--No ears found after clean up...aborting".format(fullpath)) # Log
		raise Exception	

	ears = img.copy()
	ears[filtered == 0] = 0


	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	###################################  Remove white paper ##################################
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	cnts = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cnts = cnts[0] if len(cnts) == 2 else cnts[1]
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
#Sort left to right
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b:b[1][0], reverse= False))
#Count the number of ears and number them on proof
	number_of_ears = 0
	ear_masks = []
	#array to remove white things
	white = []
	for c in cnts:
		number_of_ears = number_of_ears+1
#Create ROI and find tip
		rects = cv2.minAreaRect(c)
		width_i = int(rects[1][0])
		height_i = int(rects[1][1])
		boxs = cv2.boxPoints(rects)
		boxs = np.array(boxs, dtype="int")
		src_pts_i = boxs.astype("float32")
		dst_pts_i = np.array([[0, height_i-1],[0, 0],[width_i-1, 0],[width_i-1, height_i-1]], dtype="float32")
		M_i = cv2.getPerspectiveTransform(src_pts_i, dst_pts_i)
		ear = cv2.warpPerspective(ears, M_i, (width_i, height_i))
		n_colors = 2
		pixels = np.float32(ear.reshape(-1, 3))
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
		flags = cv2.KMEANS_RANDOM_CENTERS
		_, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
		_, counts = np.unique(labels, return_counts=True)
		dominant = palette[np.argmax(counts)]
		Red = dominant[0]
		Green = dominant[1]
		Blue = dominant[2]
		#print(Red+Green+Blue)
		if Red+Green+Blue > 685:
			white.append(number_of_ears)
		#print(white)
	log.info("[EARS]--{}--Detected {} white artifacts (qr stickr or white paper) removing...".format(filename, len(white)))
	white = [x - 1 for x in white]
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	###################################  Sort ears module ####################################
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	cnts = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cnts = cnts[0] if len(cnts) == 2 else cnts[1]
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	#print(boundingBoxes)
#Sort left to right
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b:b[1][0], reverse= False))
#Remove any white objects:
	for x in white:
		if x == 0:
			cnts = cnts[1:]
		else:
			cnts = cnts[:x] + cnts[x+1:]
#Count the number of ears and number them on proof
	number_of_ears = 0
	ear_masks = []
	for c in cnts:
		number_of_ears = number_of_ears+1
#Find centroid
		M = cv2.moments(c)
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])
#Create ROI and find tip
		rects = cv2.minAreaRect(c)
		width_i = int(rects[1][0])
		height_i = int(rects[1][1])
		boxs = cv2.boxPoints(rects)
		boxs = np.array(boxs, dtype="int")
		src_pts_i = boxs.astype("float32")
		dst_pts_i = np.array([[0, height_i-1],[0, 0],[width_i-1, 0],[width_i-1, height_i-1]], dtype="float32")
		M_i = cv2.getPerspectiveTransform(src_pts_i, dst_pts_i)
		ear = cv2.warpPerspective(ears, M_i, (width_i, height_i))

		height_i = ear.shape[0]
		width_i = ear.shape[1]
		if height_i > width_i:
			rat = round(width_i/height_i, 2)
		else:
			rat = round(height_i/width_i, 2)
			ear = cv2.rotate(ear, cv2.ROTATE_90_COUNTERCLOCKWISE) 				#This rotates the image in case it is saved vertically

		ear = cv2.copyMakeBorder(ear, 30, 30, 30, 30, cv2.BORDER_CONSTANT)
		ear_area = cv2.contourArea(c)
		hulls = cv2.convexHull(c); hull_areas = cv2.contourArea(hulls)
		ear_solidity = float(ear_area)/hull_areas	
		ear_percent = (ear_area/img_area)*100
		log.info("[EARS]--{}--Ear #{}: Min Area: {}% Aspect Ratio: {} Solidity score: {}".format(filename, number_of_ears, round(ear_percent, 3), rat, round(ear_solidity, 3)))
		#Draw the countour number on the image
		ear_masks.append(ear)
		cv2.drawContours(ears_proof, [c], -1, (134,22,245), -1)
		cv2.putText(ears_proof, "#{}".format(number_of_ears), (cX - 80, cY), cv2.FONT_HERSHEY_SIMPLEX,4.0, (255, 255, 0), 10)


	cv2.putText(ears_proof, "Found {} Ear(s)".format(number_of_ears), (int((img.shape[0]/1.5)), img.shape[0]-100), cv2.FONT_HERSHEY_SIMPLEX, 5.0, (200, 255, 255), 17)
	
	if args.debug is True:
		cv2.namedWindow('[DEBUG] [EARS] Segmentation after Filter', cv2.WINDOW_NORMAL)
		cv2.resizeWindow('[DEBUG] [EARS] Segmentation after Filter', 1000, 1000)
		cv2.imshow('[DEBUG] [EARS] Segmentation after Filter', ears_proof); cv2.waitKey(3000); cv2.destroyAllWindows()


	log.info("[EARS]--{}--Found {} Ear(s) after clean up".format(filename, number_of_ears))

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	##################################  Create Found_ears_proof  #############################
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	axis = ears_proof.shape[1]
	axis = int(axis/3)
	tall_ratio = ears_proof.shape[1]/ears_proof.shape[1]
	
	img_list = []
	img_list.append(qr_proof)
	img_list.append(color_proof)
	img_list.append(ppm_proof)

	montages = utility.build_montages(img_list, (axis, int(axis*tall_ratio)), (3, 1))
	dim = (ears_proof.shape[1], int(axis*tall_ratio))
	montages[0] = cv2.resize(montages[0], dim, interpolation = cv2.INTER_AREA)

	ears_proof = cv2.vconcat([montages[0], ears_proof])

	if args.no_proof is False or args.debug is True:											# Print proof with QR code
		cv2.namedWindow('[EARS] Ear Segmentation Proof', cv2.WINDOW_NORMAL)
		cv2.resizeWindow('[EARS] Ear Segmentation Proof', 1000, 1000)
		cv2.imshow('[EARS] Ear Segmentation Proof', ears_proof); cv2.waitKey(3000); cv2.destroyAllWindows()

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Save proofs
	if args.no_save is False:		
		destin = "{}".format(out) + "01_Proofs/"
		if not os.path.exists(destin):
			try:
				os.mkdir(destin)
			except OSError as e:
				if e.errno != errno.EEXIST:
					raise
		if args.lowres_proof is False:
			destin = "{}".format(out) + "01_Proofs/" + filename + "_proof.png"
			log.info("[EARS]--{}--Proof saved to: {}".format(filename, destin))
			cv2.imwrite(destin, ears_proof)
		else:
			destin = "{}".format(out) + "01_Proofs/" + filename + "_proof.jpg"
			log.info("[EARS]--{}--Low resolution proof saved to: {}".format(filename, destin))
			cv2.imwrite(destin, ears_proof, [int(cv2.IMWRITE_JPEG_QUALITY), 50])

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	##################################  Clean silks module  ##################################
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	
	final_ear_masks = []
	n = 1 #Counter
	for r in range(number_of_ears):
		ear = ear_masks[r]
		
		_,_,reed = cv2.split(ear)											#Split into it channel constituents
		_,reed = cv2.threshold(reed, 0, 255, cv2.THRESH_OTSU)
		reed = utility.cnctfill(reed)
		ear[reed == 0] = 0

		if args.debug is True:
			cv2.namedWindow('[DEBUG][EAR] Ear Before Clean Up', cv2.WINDOW_NORMAL)
			cv2.resizeWindow('[DEBUG][EAR] Ear Before Clean Up', 1000, 1000)
			cv2.imshow('[DEBUG][EAR] Ear Before Clean Up', ear); cv2.waitKey(3000); cv2.destroyAllWindows() 

		if args.silk_cleanup != "None" and args.silk_cleanup != "":
			log.info("[SILK]--{}--Cleaning up silks iwth custom settings".format(filename))
			delta_conv = 0.001
			conv_var = float(args.silk_cleanup[0])
			i_var = float(args.silk_cleanup[1])
			i = 1

			_,_,r = cv2.split(ear)											# Split into it channel constituents
			_,r = cv2.threshold(r, 0, 255, cv2.THRESH_OTSU)
			r = utility.cnctfill(r)
			ear[r == 0] = 0
			lab = cv2.cvtColor(ear, cv2.COLOR_BGR2LAB)
			lab[r == 0] = 0
			_,_,b_chnnl = cv2.split(lab)									# Split into it channel constituents

			convexity = find_ears.calculate_convexity(b_chnnl)

			log.info("[SILK]--{}--Ear #{}: Min delta convexity: {}, Max interations: {}".format(filename, n, conv_var, 3, i_var))
						
			while delta_conv < conv_var  and i <= i_var:
				b_chnnl = cv2.morphologyEx(b_chnnl, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2+i,2+i)    ), iterations=1+i) #Open to get rid of the noise
				convexity2 = find_ears.calculate_convexity(b_chnnl)
				delta_conv = convexity2-convexity
				log.info("[SILK]--{}--Ear #{}: Convexity: {}, delta convexity: {}, iteration: {}".format(filename, n, round(convexity2, 3), round(delta_conv, 3), i))
				i = i + 1
				
			ear[b_chnnl == 0] = 0
			log.info("[SILK]--{}--Silk clean-up module finished. Final convexity--{}".format(filename, round(convexity2, 3)))
			ear = cv2.copyMakeBorder(ear, 1000, 1000, 1000, 1000, cv2.BORDER_CONSTANT)
			_,_,ear_binary = cv2.split(ear)											#Split into it channel constituents
			_,ear_binary = cv2.threshold(ear_binary, 0, 255, cv2.THRESH_OTSU)
			ear_binary = utility.cnctfill(ear_binary)
			cnts = cv2.findContours(ear_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cnts = cnts[0] if len(cnts) == 2 else cnts[1]
			for c in cnts:
				#Create ROI and find tip
				rects = cv2.minAreaRect(c)
				width_i = int(rects[1][0])
				height_i = int(rects[1][1])
				boxs = cv2.boxPoints(rects)
				boxs = np.array(boxs, dtype="int")
				src_pts_i = boxs.astype("float32")
				dst_pts_i = np.array([[0, height_i-1],[0, 0],[width_i-1, 0],[width_i-1, height_i-1]], dtype="float32")
				M_i = cv2.getPerspectiveTransform(src_pts_i, dst_pts_i)
				ear = cv2.warpPerspective(ear, M_i, (width_i, height_i))
				ear = cv2.copyMakeBorder(ear, 30, 30, 30, 30, cv2.BORDER_CONSTANT)
				height_i = ear.shape[0]
				width_i = ear.shape[1]
				if height_i > width_i:
					rat = round(width_i/height_i, 2)
				else:
					rat = round(height_i/width_i, 2)
					ear = cv2.rotate(ear, cv2.ROTATE_90_COUNTERCLOCKWISE) 				#This rotates the image in case it is saved vertically

		elif args.silk_cleanup != "None":
			_,_,r = cv2.split(ear)											# Split into it channel constituents
			_,r = cv2.threshold(r, 0, 255, cv2.THRESH_OTSU)
			r = utility.cnctfill(r)
			ear[r == 0] = 0
			lab = cv2.cvtColor(ear, cv2.COLOR_BGR2LAB)
			lab[r == 0] = 0
			_,_,b_chnnl = cv2.split(lab)									# Split into it channel constituents

			convexity = find_ears.calculate_convexity(b_chnnl)
			
			default_silk_convexity = 0.87
			log.info("[SILK]--{}--Silk clean-up module turned on with default silk convexity threshold of {}".format(filename, default_silk_convexity))
				
			if convexity < default_silk_convexity:
				log.warning("[SILK]--{}--Ear #{}: Convexity under {} has triggered default ear clean-up module".format(filename, n, default_silk_convexity))
				conv_var = 0.04
				i_var = 10
				i = 1
				delta_conv = 0.001
				log.info("[SILK]--{}--Ear #{}: Min delta convexity: {}, Max interations: {}".format(filename, n, conv_var, 3, i_var))
				while delta_conv < conv_var  and i <= i_var:
					b_chnnl = cv2.morphologyEx(b_chnnl, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2+i,2+i)    ), iterations=1+i) #Open to get rid of the noise
					convexity2 = find_ears.calculate_convexity(b_chnnl)
					delta_conv = convexity2-convexity
					log.info("[SILK]--{}--Ear #{}: Convexity: {}, delta convexity: {}, iteration: {}".format(filename, n, round(convexity2, 3), round(delta_conv, 3), i))
					i = i + 1
				ear[b_chnnl == 0] = 0
				log.info("[SILK]--{}--Silk clean-up module finished. Final convexity--{}".format(filename, round(convexity2, 3)))

				ear = cv2.copyMakeBorder(ear, 1000, 1000, 1000, 1000, cv2.BORDER_CONSTANT)
				_,_,ear_binary = cv2.split(ear)											#Split into it channel constituents
				_,ear_binary = cv2.threshold(ear_binary, 0, 255, cv2.THRESH_OTSU)
				ear_binary = utility.cnctfill(ear_binary)
				cnts = cv2.findContours(ear_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cnts = cnts[0] if len(cnts) == 2 else cnts[1]
				for c in cnts:
					#Create ROI and find tip
					rects = cv2.minAreaRect(c)
					width_i = int(rects[1][0])
					height_i = int(rects[1][1])
					boxs = cv2.boxPoints(rects)
					boxs = np.array(boxs, dtype="int")
					src_pts_i = boxs.astype("float32")
					dst_pts_i = np.array([[0, height_i-1],[0, 0],[width_i-1, 0],[width_i-1, height_i-1]], dtype="float32")
					M_i = cv2.getPerspectiveTransform(src_pts_i, dst_pts_i)
					ear = cv2.warpPerspective(ear, M_i, (width_i, height_i))
					ear = cv2.copyMakeBorder(ear, 30, 30, 30, 30, cv2.BORDER_CONSTANT)
					
					height_i = ear.shape[0]
					width_i = ear.shape[1]
					if height_i > width_i:
						rat = round(width_i/height_i, 2)
					else:
						rat = round(height_i/width_i, 2)
						ear = cv2.rotate(ear, cv2.ROTATE_90_COUNTERCLOCKWISE) 				#This rotates the image in case it is saved vertically
			else:
				log.info("[SILK]--{}--Silk convexity {} under threshold {}. Ear clean-up module turned off.".format(filename,convexity,default_silk_convexity))
		else:
			log.info("[SILK]--{}--Silk clean-up module turned off.".format(filename))			

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	##################################  Orient ears module  ##################################
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
		if args.rotation is True:
			log.info('[EAR]--{}--Ear #{}: Checking ear orientation...'.format(filename, n))
			ori_width = find_ears.rotate_ear(ear)
			if ori_width[2] < ori_width[0]:
				log.warning('[EAR]--{}--Ear #{}: Ear rotated'.format(filename, n))
				ear = cv2.rotate(ear, cv2.ROTATE_180)	
			else:
				log.info('[EAR]--{}--Ear #{}: Ear orientation is fine.'.format(filename, n))
		else:
			log.info('[EAR]--{}--Ear #{}: Ear orientation turned off.'.format(filename, n))	

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	##################################  Save final ear masks  ################################
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
		#create transparency?
		tmp = cv2.cvtColor(ear, cv2.COLOR_BGR2GRAY)
		_,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
		b, g, r = cv2.split(ear)
		rgba = [b,g,r, alpha]
		ear_trans = cv2.merge(rgba,4)
		
		if args.no_save is False:
			destin = "{}".format(out) + "02_Ear_ROIs/"
			if not os.path.exists(destin):
				try:
					os.mkdir(destin)
				except OSError as e:
					if e.errno != errno.EEXIST:
						raise
			destin = "{}02_Ear_ROIs/{}_ear_{}".format(out, filename, n) + ".png"
			log.info("[EAR]--{}--Ear #{}: ROI saved to: {}".format(filename, n, destin))			
			cv2.imwrite(destin, ear_trans)

		if args.debug is True:
			cv2.namedWindow('[DEBUG][EAR] After Silk, Clean Up, and Rot', cv2.WINDOW_NORMAL)
			cv2.resizeWindow('[DEBUG][EAR] After Silk, Clean Up, and Rot', 1000, 1000)
			cv2.imshow('[DEBUG][EAR] After Silk, Clean Up, and Rot', ear); cv2.waitKey(3000); cv2.destroyAllWindows() 

		final_ear_masks.append(ear)
		n = n + 1

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	############################# BASIC FULL EAR FEATURES ####################################
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	log.info('[EAR]--{}--Extracting features from {} ears...'.format(filename, number_of_ears))
	n = 1 #Counter
	for r in range(number_of_ears):
		ear = final_ear_masks[r]
		log.info('[EAR]--{}--Ear #{}: Extracting basic morphological features...'.format(filename, n))
		#Ear_Area = Convexity = Solidity = Ellipse = Ear_Box_Width = Ear_Box_Length = Ear_Box_Area = Ear_Extreme_Length = Ear_Area_DP = Solidity_PolyDP = Solidity_Box = Taper_PolyDP = Taper_Box = Widths = Widths_Sdev = Cents_Sdev = Ear_area = Tip_Area = Bottom_Area = Krnl_Area = Tip_Fill = Blue = Green = Red = Hue = Sat = Vol = Light = A_chnnl = B_chnnl = second_width = mom1 = None
		Ear_Area, Ear_Box_Area, Ear_Box_Length, Ear_Extreme_Length, Ear_Box_Width, newWidths, max_Width, MA, ma, perimeters, Convexity, Solidity, Convexity_polyDP, Taper, Taper_Convexity, Taper_Solidity, Taper_Convexity_polyDP, Widths_Sdev, Cents_Sdev, ear_proof, canvas, wid_proof = features.extract_feats(ear, PixelsPerMetric)
		log.info('[EAR]--{}--Ear #{}: Done extracting basic morphological features'.format(filename, n))

		moments = features.extract_moments(ear)
		log.info('[EAR]--{}--Ear #{}: Done extracting image moments'.format(filename, n))


	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	############################# Cob Segemntation Module ####################################
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
		_,_,red = cv2.split(ear)											#Split into it channel constituents
		_,red = cv2.threshold(red, 0, 255, cv2.THRESH_OTSU)
		hsv = cv2.cvtColor(ear, cv2.COLOR_BGR2HSV)						#Convert into HSV color Space	
		hsv[red == 0] = 0
		h,s,_ = cv2.split(hsv)
		otsu_s,_ = cv2.threshold(s[s !=  0],1,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU); #spcial case diagnostic

		tip = np.zeros_like(red)
		bottom = np.zeros_like(red)
		red_tip = red.copy()
		tip_test = red.copy()
		red_bottom = red.copy()
		bottom_test = red.copy()

		if args.debug is True:
			cv2.namedWindow('[DEBUG][EAR] Tip Thresholding Channel Hue', cv2.WINDOW_NORMAL)
			cv2.resizeWindow('[DEBUG][EAR] Tip Thresholding Channel Hue', 1000, 1000)
			cv2.imshow('[DEBUG][EAR] Tip Thresholding Channel Hue', h); cv2.waitKey(3000); cv2.destroyAllWindows()
			cv2.namedWindow('[DEBUG][EAR] Tip Thresholding Channel Sat', cv2.WINDOW_NORMAL)
			cv2.resizeWindow('[DEBUG][EAR] Tip Thresholding Channel Sat', 1000, 1000)
			cv2.imshow('[DEBUG][EAR] Tip Thresholding Channel Sat', s); cv2.waitKey(3000); cv2.destroyAllWindows()			

		### TIP SEGMENTATION		
		if args.tip is not None:		
			if args.tip == []:
				if otsu_s < 70:
					chnnl=cv2.bitwise_not(h)
					_, otsu = cob_seg.otsu(chnnl)
					tip_thresh_int = 1					
					tip = cob_seg.manual(chnnl, otsu*tip_thresh_int)
					log.warning("[EAR]--{}--Ear #{}: Detected white ear {}...thresholding ear tip with hue channel...".format(filename, n, otsu_s))
					log.info("[EAR]--{}--Ear #{}: Segmenting ear tip with adaptive otsu approach on hue channel at intensitiy of {}...score {}".format(filename, n, tip_thresh_int, (otsu*tip_thresh_int)))
					log.info("[EAR]--{}--Ear #{}: Otsu found {} threshold".format(filename, n, otsu))
					tip = cv2.bitwise_not(tip) #invert
					if args.debug is True:
						cv2.namedWindow('[DEBUG][EAR] Tip Thresholding', cv2.WINDOW_NORMAL)
						cv2.resizeWindow('[DEBUG][EAR] Tip Thresholding', 1000, 1000)
						cv2.imshow('[DEBUG][EAR] Tip Thresholding', tip); cv2.waitKey(3000); cv2.destroyAllWindows() 

					tip_percent = 35
					dialate = 1
					tip = cob_seg.top_modifier(ear, tip, tip_percent, dialate, args.debug)	
					tip_test = cob_seg.top_modifier(ear, red_tip, tip_percent, dialate,False)
					log.info("[EAR]--{}--Ear #{}: Processing tip with {} tip percent, {} dialate".format(filename, n, tip_percent, dialate))				

				else:
					chnnl = s.copy()
					_, otsu = cob_seg.otsu(chnnl)
					tip_thresh_int = 1.4					
					tip = cob_seg.manual(chnnl, otsu*tip_thresh_int)
					log.info("[EAR]--{}--Ear #{}: Segmenting ear tip with adaptive otsu approach on saturation channel at intensitiy of {}...score {}".format(filename, n, tip_thresh_int, otsu*tip_thresh_int))
					log.info("[EAR]--{}--Ear #{}: Otsu found {} threshold".format(filename, n, otsu))
					tip = cv2.bitwise_not(tip) #invert			
					if args.debug is True:
						cv2.namedWindow('[DEBUG][EAR] Tip Thresholding', cv2.WINDOW_NORMAL)
						cv2.resizeWindow('[DEBUG][EAR] Tip Thresholding', 1000, 1000)
						cv2.imshow('[DEBUG][EAR] Tip Thresholding', tip); cv2.waitKey(3000); cv2.destroyAllWindows() 
					tip_percent = 35
					dialate = 1
					tip = cob_seg.top_modifier(ear, tip, tip_percent, dialate, args.debug)	
					tip_test = cob_seg.top_modifier(ear, red_tip, tip_percent, dialate, False)
					log.info("[EAR]--{}--Ear #{}: Processing tip with {} tip percent, {} dialate".format(filename, n, tip_percent, dialate))	
			else:
				if args.tip[0] == 'h':
					chnnl=cv2.bitwise_not(h)
				else:
					args.tip[0] == 's'
					chnnl = s.copy()
				_, otsu = cob_seg.otsu(chnnl)
				tip_thresh_int = float(args.tip[1])
				tip = cob_seg.manual(chnnl, otsu*tip_thresh_int)
				log.info("[EAR]--{}--Ear #{}: Segmenting ear tip with custom thresholding in {} channel at intensitiy of {}...score {}".format(filename, n, args.tip[0], args.tip[1], (otsu*tip_thresh_int)))
				tip = cv2.bitwise_not(tip) #invert
				if args.debug is True:
					cv2.namedWindow('[DEBUG][EAR] Tip Thresholding', cv2.WINDOW_NORMAL)
					cv2.resizeWindow('[DEBUG][EAR] Tip Thresholding', 1000, 1000)
					cv2.imshow('[DEBUG][EAR] Tip Thresholding', tip); cv2.waitKey(3000); cv2.destroyAllWindows() 
				tip_percent = args.tip[2]
				dialate = args.tip[3]
				tip = cob_seg.top_modifier(ear, tip, tip_percent, dialate, args.debug)	
				tip_test = cob_seg.top_modifier(ear, red_tip, 50, dialate, False)
				log.info("[EAR]--{}--Ear #{}: Processing tip with custom settings: {} tip percent, {} dialate".format(filename, n, tip_percent, dialate))	

		else:
			log.info("[EAR]--{}--Ear #{}: Ear tip segmentation turned off".format(filename, n))

		### BOTTOM SEGMENTATION		
		if args.bottom is not None:		
			if args.bottom == []:
				if otsu_s < 70:
					chnnl=cv2.bitwise_not(h)
					_, otsu = cob_seg.otsu(chnnl)
					bottom_thresh_int = 1
					bottom = cob_seg.manual(chnnl, otsu*bottom_thresh_int)
					log.warning("[EAR]--{}--Ear #{}: Detected white ear {}...thresholding ear bottom with hue channel...".format(filename, n, otsu_s))
					log.info("[EAR]--{}--Ear #{}: Segmenting ear bottom with adaptive otsu approach on hue channel with default intensity {}...score {}".format(filename, n, bottom_thresh_int, (otsu*bottom_thresh_int)))
					log.info("[EAR]--{}--Ear #{}: Otsu found {} threshold".format(filename, n, otsu))

					bottom = cv2.bitwise_not(bottom) #invert
					if args.debug is True:
						cv2.namedWindow('[DEBUG][EAR] bottom Thresholding', cv2.WINDOW_NORMAL)
						cv2.resizeWindow('[DEBUG][EAR] bottom Thresholding', 1000, 1000)
						cv2.imshow('[DEBUG][EAR] bottom Thresholding', bottom); cv2.waitKey(3000); cv2.destroyAllWindows() 

					bottom_percent = 85
					dialate = 1
					bottom = cob_seg.bottom_modifier(ear, bottom, bottom_percent, dialate, args.debug)
					bottom_test = cob_seg.bottom_modifier(ear, red_bottom, bottom_percent, dialate, False)
					log.info("[EAR]--{}--Ear #{}: Processing bottom with {} bottom percent, {} dialate".format(filename, n, bottom_percent, dialate))
				else:
					chnnl = s.copy()
					_, otsu = cob_seg.otsu(chnnl)
					bottom_thresh_int = 1.4
					bottom = cob_seg.manual(chnnl, otsu*bottom_thresh_int)
					log.info("[EAR]--{}--Ear #{}: Segmenting ear bottom with adaptive otsu approach on saturation channel with default intensity {}...score {}".format(filename, n, bottom_thresh_int, (otsu*bottom_thresh_int)))
					log.info("[EAR]--{}--Ear #{}: Otsu found {} threshold".format(filename, n, otsu))
					bottom = cv2.bitwise_not(bottom) #invert
					if args.debug is True:
						cv2.namedWindow('[DEBUG][EAR] bottom Thresholding', cv2.WINDOW_NORMAL)
						cv2.resizeWindow('[DEBUG][EAR] bottom Thresholding', 1000, 1000)
						cv2.imshow('[DEBUG][EAR] bottom Thresholding', bottom); cv2.waitKey(3000); cv2.destroyAllWindows() 

					bottom_percent = 85
					dialate = 1
					bottom = cob_seg.bottom_modifier(ear, bottom, bottom_percent, dialate, args.debug)	
					bottom_test = cob_seg.bottom_modifier(ear, red_bottom, bottom_percent, dialate, False)
					log.info("[EAR]--{}--Ear #{}: Processing bottom with {} bottom percent, {} dialate,".format(filename, n, bottom_percent, dialate))
			else:
				if args.bottom[0] == 'h':
					chnnl=cv2.bitwise_not(h)
				elif args.bottom[0] == 's':
					chnnl = s.copy()
				_, otsu = cob_seg.otsu(chnnl)
				bottom_thresh_int = float(args.bottom[1])
				bottom = cob_seg.manual(chnnl, otsu*bottom_thresh_int)
				log.info("[EAR]--{}--Ear #{}: Segmenting ear bottom with custom thresholding in {} channel at intensitiy of {}...score {}".format(filename, n, args.bottom[0], args.bottom[1], (otsu*bottom_thresh_int)))
				bottom = cv2.bitwise_not(bottom) #invert
				if args.debug is True:
					cv2.namedWindow('[DEBUG][EAR] bottom Thresholding', cv2.WINDOW_NORMAL)
					cv2.resizeWindow('[DEBUG][EAR] bottom Thresholding', 1000, 1000)
					cv2.imshow('[DEBUG][EAR] bottom Thresholding', bottom); cv2.waitKey(3000); cv2.destroyAllWindows() 

				bottom_percent = args.bottom[2]
				dialate = args.bottom[3]
				bottom = cob_seg.bottom_modifier(ear, bottom, bottom_percent, dialate, args.debug)	
				bottom_test = cob_seg.bottom_modifier(ear, red_bottom, 80, dialate, False)
				log.info("[EAR]--{}--Ear #{}: Processing bottom with custom settings: {} bottom percent, {} dialate".format(filename, n, bottom_percent, dialate))					
		else:
			log.info("[EAR]--{}--Ear #{}: Ear bottom segmentation turned off".format(filename, n))

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	############################# Cob/shank/kernel Analysis Module ###########################
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
		if cv2.countNonZero(tip_test) > 0:
			if (cv2.countNonZero(tip)/cv2.countNonZero(tip_test)) > 0.999 :
				tip = np.zeros_like(red)

		if cv2.countNonZero(bottom_test) > 0:
			if (cv2.countNonZero(bottom)/cv2.countNonZero(bottom_test)) > 0.999:
				bottom = np.zeros_like(red)

		log.info("[EAR]--{}--Ear #{}: Extracting kernel features...".format(filename, n))
		Tip_Area, Bottom_Area, Krnl_Area, Kernel_Length, Krnl_Convexity, Tip_Fill, Bottom_Fill, Krnl_Fill, krnl_proof, cob, uncob, Blue, Red, Green, Hue, Sat, Vol, Light, A_chnnl, B_chnnl = features.krnl_feats(ear, tip, bottom, PixelsPerMetric)
		log.info("[EAR]--{}--Ear #{}: Done extracting kernel features".format(filename, n))
			
		Krnl_proof = ear.copy()
		Krnl_proof[cob == 255] = [0,0,255]

		if args.debug is True:
			cv2.namedWindow('[DEBUG][EAR] Cob Segmentation', cv2.WINDOW_NORMAL)
			cv2.resizeWindow('[DEBUG][EAR] Cob Segmentation', 1000, 1000)
			cv2.imshow('[DEBUG][EAR] Cob Segmentation', Krnl_proof); cv2.waitKey(3000); cv2.destroyAllWindows() 

			cv2.namedWindow('[DEBUG][EAR] Dominant Color', cv2.WINDOW_NORMAL)
			cv2.resizeWindow('[DEBUG][EAR] Dominant Color', 1000, 1000)
			cv2.imshow('[DEBUG][EAR] Dominant Color', krnl_proof); cv2.waitKey(3000); cv2.destroyAllWindows() 

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	################################## KRN Module ###########################################
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
		if args.kernel_row_number is True:
			log.info("[EAR]--{}--Ear #{}: KRN module turned on. Extracting number of kernel peaks and kernel width...".format(filename, n))
			global_peak_num, global_mean_diff, global_diff_sd, krn_proof = krn.krn(ear, args.debug)

			if args.debug is True:
				cv2.namedWindow('[DEBUG][EAR] Cob Segmentation', cv2.WINDOW_NORMAL)
				cv2.resizeWindow('[DEBUG][EAR] Cob Segmentation', 1000, 1000)
				cv2.imshow('[DEBUG][EAR] Cob Segmentation', krn_proof); cv2.waitKey(3000); cv2.destroyAllWindows() 

			KRN_Boundaries = global_peak_num
			KRN_Std_Dev = global_diff_sd

			if PixelsPerMetric is not None:
				global_mean_diff = global_mean_diff / (PixelsPerMetric)

			Mean_Kernel_Width = global_mean_diff

			log.info("[EAR]--{}--Ear #{}: Using kernel width and radius to predict KRN...".format(filename, n))

			inside_mean, centa_mean, areasec_mean, areacirc_mean, KRN = utility.circ((Ear_Box_Width/2), global_mean_diff)
			log.info("[EAR]--{}--Ear #{}: KRN analysis done.".format(filename, n))
		else:
			KRN = None
			Mean_Kernel_Width = None
			KRN_Boundaries = None
			KRN_Std_Dev = None
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	################################## Grading Module ########################################
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
		if args.usda_grade is True:
			log.info("[EAR]--{}--Ear #{}: Using ear features to determine USDA grade...(requires ppm flag)".format(filename, n))
			if args.pixelspermetric[1] == 'cm':
				Facy_Len = 12.7
				No1_Len = 10.16
			elif args.pixelspermetric[1] == 'in':
				Facy_Len = 5
				No1_Len = 4

			#Len
			if Ear_Box_Length >= Facy_Len:
				USDA_Grade_Len = "Fancy"
			elif Ear_Box_Length >= No1_Len:
				USDA_Grade_Len = "No.1"
			else:
				USDA_Grade_Len = "Off Grade"

			#Fill #this technically should be length ~~~ratio~~~
			if Tip_Fill >= 0.875:
				USDA_Grade_Fill = "Well Filled"
			elif Tip_Fill >= 0.792:
				USDA_Grade_Fill = "Moderately Filled"
			else:
				USDA_Grade_Fill = "Poorly Filled"
			log.info("[EAR]--{}--Ear #{}: USDA graded.".format(filename, n))
		else:
			log.info("[EAR]--{}--Ear #{}: USDA grading turned off.".format(filename, n))
			USDA_Grade_Len = None
			USDA_Grade_Fill = None
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	##################################### Proofs  ############################################
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
		canvas = cv2.cvtColor(canvas,cv2.COLOR_GRAY2RGB)
		ear_proof = cv2.hconcat([canvas, ear_proof, wid_proof, Krnl_proof, krnl_proof])

		if args.kernel_row_number is True:
			dim = (krnl_proof.shape[1], krnl_proof.shape[0])
			krn_proof = cv2.resize(krn_proof, dim, interpolation = cv2.INTER_AREA)
			ear_proof = cv2.hconcat([ear_proof, krn_proof])

		if args.no_save is False:
			destin = "{}".format(out) + "03_Ear_Proofs/"
			if not os.path.exists(destin):
				try:
					os.mkdir(destin)
				except OSError as e:
					if e.errno != errno.EEXIST:
						raise
			if args.lowres_proof is False:			
				destin = "{}03_Ear_Proofs/{}_ear_{}_proof".format(out, filename, n) + ".png"
				log.info("[EAR]--{}--Ear #{}: Ear proof saved to: {}".format(filename, n, destin))			
				cv2.imwrite(destin, ear_proof)
			else:
				destin = "{}03_Ear_Proofs/{}_ear_{}_proof".format(out, filename, n) + ".jpg"
				log.info("[EAR]--{}--Ear #{}: Low resolution ear proof saved to: {}".format(filename, n, destin))			
				cv2.imwrite(destin, ear_proof, [int(cv2.IMWRITE_JPEG_QUALITY), 50])



		if args.no_proof is False or args.debug is True:
			cv2.namedWindow('[Ear] Final Proof', cv2.WINDOW_NORMAL)
			cv2.resizeWindow('[Ear] Final Proof', 1000, 1000)
			cv2.imshow('[Ear] Final Proof', ear_proof); cv2.waitKey(3000); cv2.destroyAllWindows()

		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
		##################################### Features CSV  ######################################
		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Create full feature CSV
		if args.no_save is False:			
			csvname = out + 'all_features' +'.csv'
			file_exists = os.path.isfile(csvname)
			with open (csvname, 'a') as csvfile:
				headers = ['Filename', 'PixelsPerMetric', 'Units', 'Ear Number', 'Ear_Area', 'Ear_Box_Area', 'Ear_Box_Length', 'Ear_Extreme_Length', 'Ear_Box_Width',
							'Max_Width', 'MA_Ellipse', 'ma_Ellipse', 'Perimeter', 'Convexity', 'Solidity', 'Convexity_polyDP', 'Taper',
							'Taper_Convexity', 'Taper_Solidity', 'Taper_Convexity_polyDP', 'Widths_Sdev', 'Curvature', 'Tip_Area', 'Bottom_Area',
							'Krnl_Area', 'Kernel_Length', 'Krnl_Convexity', 'Tip_Fill', 'Bottom_Fill', 'Krnl_Fill', 'KRN_Pred', 'Mean_Kernel_Width',
							'KRN_Boundaries', 'KRN_Std_Dev', 'USDA_Grade_Len', 'USDA_Grade_Fill' ,'Blue', 'Red', 'Green', 'Hue', 'Sat', 'Vol', 'Light',
							'A_chnnl', 'B_chnnl', 'm00','m10','m01','m20','m11','m02','m30','m21','m12','m03','mu20','mu11','mu02','mu30','mu21',
							'mu12','mu03','nu20','nu11','nu02','nu30','nu21','nu12','nu03'] 

				writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n',fieldnames=headers)
				if not file_exists:
					writer.writeheader()  # file doesn't exist yet, write a header	

				writer.writerow({'Filename': filename, 'PixelsPerMetric': PixelsPerMetric, 'Units': Units, 'Ear Number': n, 'Ear_Area': Ear_Area, 'Ear_Box_Area': Ear_Box_Area,
								 'Ear_Box_Length': Ear_Box_Length, 'Ear_Extreme_Length': Ear_Extreme_Length, 'Ear_Box_Width': Ear_Box_Width,
								 'Max_Width': max_Width, 'MA_Ellipse': MA, 'ma_Ellipse': ma, 'Perimeter': perimeters,
								 'Convexity': Convexity , 'Solidity': Solidity, 'Convexity_polyDP': Convexity_polyDP, 'Taper': Taper,
								 'Taper_Convexity': Taper_Convexity, 'Taper_Solidity': Taper_Solidity, 'Taper_Convexity_polyDP': Taper_Convexity_polyDP, 
							     'Widths_Sdev': Widths_Sdev, 'Curvature': Cents_Sdev, 'Tip_Area': Tip_Area, 'Bottom_Area': Bottom_Area, 
							     'Krnl_Area': Krnl_Area, 'Kernel_Length': Kernel_Length , 'Krnl_Convexity': Krnl_Convexity, 'Tip_Fill': Tip_Fill, 
								 'Bottom_Fill': Bottom_Fill, 'Krnl_Fill': Krnl_Fill , 'KRN_Pred': KRN, 'KRN_Boundaries': KRN_Boundaries, 'Mean_Kernel_Width': Mean_Kernel_Width,
								 'KRN_Std_Dev': KRN_Std_Dev, 'USDA_Grade_Len': USDA_Grade_Len, 'USDA_Grade_Fill': USDA_Grade_Fill, 'Blue': Blue , 'Red': Red , 'Green': Green , 'Hue': Hue, 'Sat': Sat,
								 'Vol': Vol , 'Light': Light , 'A_chnnl': A_chnnl , 'B_chnnl': B_chnnl, 'm00': moments['m00'],'m10': moments['m10'],'m01': moments['m01'],'m20': moments['m20'],'m11': moments['m11'],
								 'm02': moments['m02'],'m30': moments['m30'],'m21': moments['m21'],'m12': moments['m12'],'m03': moments['m03'],'mu20': moments['mu20'],
								 'mu11': moments['mu11'],'mu02': moments['mu02'],'mu30': moments['mu30'],'mu21': moments['mu21'],'mu12': moments['mu12'],'mu03': moments['mu03'],
								 'nu20': moments['nu20'],'nu11': moments['nu11'],'nu02': moments['nu02'],'nu30': moments['nu30'],'nu21': moments['nu21'],'nu12': moments['nu12'],'nu03': moments['nu03']})

			log.info("[EAR]--{}--Ear #{}: Saved all features to: {}all_features.csv".format(filename, n, out))

		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Create Condensed Featureset
		if args.no_save is False:
			csvname = out + 'features_condensed' +'.csv'
			file_exists = os.path.isfile(csvname)
			with open (csvname, 'a') as csvfile:
				headers = ['Filename', 'PixelsPerMetric', 'Units', 'QR_Code', 'Ear_Number', 'Ear_Area', 
							'Ear_Length', 'Ear_Width','Solidity','Convexity_polyDP','Taper',
							'Curvature', 'Krnl_Area', 'Tip_Fill', 'Predicted_KRN']

				writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n',fieldnames=headers)
				if not file_exists:
					writer.writeheader()  # file doesn't exist yet, write a header

				writer.writerow({'Filename': filename, 'PixelsPerMetric': PixelsPerMetric, 'Units': Units, 'QR_Code': QRcodeData , 'Ear_Number': n, 'Ear_Area': Ear_Area, 'Ear_Length': Ear_Extreme_Length, 'Ear_Width': max_Width,
								 'Solidity': Solidity, 'Convexity_polyDP': Convexity_polyDP, 'Taper': Taper_Convexity_polyDP, 
							     'Curvature': Widths_Sdev, 'Krnl_Area': Krnl_Area, 'Tip_Fill': Krnl_Fill , 'Predicted_KRN': KRN})

			log.info("[EAR]--{}--Ear #{}: Saved condensed features to: {}features_condensed.csv".format(filename, n, out))

		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Create KRN CSV tp run XGBOOST
		if args.no_save is False:			
			csvname = out + 'krn_features' +'.csv'
			file_exists = os.path.isfile(csvname)
			with open (csvname, 'a') as csvfile:
				headers = ['Filename', 'Units', 'Ear Number', 'Ear_Area', 'Ear_Extreme_Length', 'Max_Width', 'Solidity','Convexity_polyDP',
							'Taper_Convexity_polyDP', 'Widths_Sdev', 'Krnl_Area', 'Krnl_Fill', 'KRN_Pred', 'KRN_Boundaries',
							 'Mean_Kernel_Width', 'KRN_Std_Dev', 'm00','m10','m01','m20','m11','m02','m30','m21','m12','m03','mu20','mu11','mu02','mu30','mu21','mu12','mu03','nu20',
							'nu11','nu02','nu30','nu21','nu12','nu03']


				writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n',fieldnames=headers)
				if not file_exists:
					writer.writeheader()  # file doesn't exist yet, write a header	


				writer.writerow({'Filename': filename, 'Units': Units, 'Ear Number': n, 'Ear_Area': Ear_Area, 'Ear_Extreme_Length': Ear_Extreme_Length, 'Max_Width': max_Width,
								 'Solidity': Solidity, 'Convexity_polyDP': Convexity_polyDP, 'Taper_Convexity_polyDP': Taper_Convexity_polyDP, 
							     'Widths_Sdev': Widths_Sdev, 'Krnl_Area': Krnl_Area, 'Krnl_Fill': Krnl_Fill , 'KRN_Pred': KRN, 'KRN_Boundaries': KRN_Boundaries, 'Mean_Kernel_Width': Mean_Kernel_Width,
								 'KRN_Std_Dev': KRN_Std_Dev, 'm00': moments['m00'],'m10': moments['m10'],'m01': moments['m01'],'m20': moments['m20'],'m11': moments['m11'],
								 'm02': moments['m02'],'m30': moments['m30'],'m21': moments['m21'],'m12': moments['m12'],'m03': moments['m03'],'mu20': moments['mu20'],
								 'mu11': moments['mu11'],'mu02': moments['mu02'],'mu30': moments['mu30'],'mu21': moments['mu21'],'mu12': moments['mu12'],'mu03': moments['mu03'],
								 'nu20': moments['nu20'],'nu11': moments['nu11'],'nu02': moments['nu02'],'nu30': moments['nu30'],'nu21': moments['nu21'],'nu12': moments['nu12'],'nu03': moments['nu03']})

			log.info("[EAR]--{}--Ear #{}: Saved kernel features to: {}krn_features.csv".format(filename, n, out))

		n = n + 1
	log.info("[EAR]--{}--Collected all ear features.".format(filename))		

#def run():
#    """Entry point for console_scripts"""
#    main(sys.argv[1:])


#if __name__ == "__main__":
#    run()
main()