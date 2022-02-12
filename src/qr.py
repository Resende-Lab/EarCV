# -*- coding: utf-8 -*-
"""
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
##################################  QR CODE MODULE  ######################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

This tool scans image for QR code and extracts information using pyzbar's decode function.
This tool can optionally break the image into sublocks with used defined window and overlap. 
	Credit to: Devyanshu, https://github.com/Devyanshu/image-split-with-overlap
This script requires that `OpenCV 2', 'numpy', and 'pyzbar' be installed within the Python environment you are running this script in.
This script imports the utility.py module within the same folder.

"""
import cv2
import sys
import numpy as np
import utility
import find_ears
from pyzbar.pyzbar import decode


#for the csv generator
import traceback
import os
import re
import string
import csv


def qr_scan(qr_img, qr_window_size, overlap, debug):
	"""
	Scans image for QR code and extracts information using pyzbar's decode function.

	Parameters
	----------
	qr_img : array_like
		Valid file path to image to be scanned for QR code. Accepted formats: 'tiff', 'jpeg', 'bmp', 'png'.


	qr_window_size: float
		Optional. Dimension of square window size to scan over original image.

	overlap: float
		Optional. Amount of overlap between windows. Must be a decimal between 0 & 1. The higher the number the more overlap between windows and higher scanning resolution but longer analysis.

	debug: bool
		If true, print images.

	Returns
	-------
	QRcodeType
	QRcodeData
	QRcodeRect
	qr_count
	qr_proof

	References
	----------

	Thank you zbar! http://zbar.sourceforge.net/index.html

	Examples
	--------

	Example 1:

	python qr.py W201432.JPG None None False

	Example 2:

	python qr.py W201432.JPG 2000 0.01 True

	"""
	

	id = []
	qr_count = 1
	QRcodeType = QRcodeData = QRcodeRect = qr_proof = removesticker =None
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Scan entire image for QRcode
	if qr_window_size is None:

		#qr_img = cv2.cvtColor(qr_img,cv2.COLOR_BGR2GRAY)
		#_,mask = cv2.threshold(qr_img, 0, 255, cv2.THRESH_OTSU)
		#mask = cv2.inRange(qr_img,(0,0,0),(256,256,256)) ###mistake to hard code these

		#try a few things here
		b,g,r = cv2.split(qr_img)		
		#qr_img = cv2.cvtColor(qr_img,cv2.COLOR_BGR2GRAY)
		otsu,_ = cv2.threshold(g, 0, 255, cv2.THRESH_OTSU)
		mask = cv2.threshold(g, int(otsu*1.30),256, cv2.THRESH_BINARY)[1]

		thresholded = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
		inverted = thresholded
		#inverted = 255-thresholded # black-in-white
		if debug is True:
			cv2.namedWindow('[DEBUG] [QR] Scanning QRcode', cv2.WINDOW_NORMAL)
			cv2.resizeWindow('[DEBUG] [QR] Scanning QRcode', 1000, 1000)
			cv2.imshow('[DEBUG] [QR] Scanning QRcode', inverted); cv2.waitKey(2000); cv2.destroyAllWindows()

		id = decode(inverted)
		if id != []:
			qr_proof = inverted
			for QRcode in decode(inverted):
				id= QRcode.data.decode()
				QRcodeRect = QRcode.rect
				QRcodeData = QRcode.data.decode("utf-8")
				QRcodeType = QRcode.type
			#try to remove sticker 
			(x, y, w, h) = QRcodeRect
			removesticker = np.zeros_like(mask)
			qr_center_y = y+int(w/2)
			qr_center_x = x+int(h/2)

			#revisit this!!!!!
			removesticker = cv2.rectangle(removesticker, (qr_center_x-int(3*h),qr_center_y-int(3*h)), (qr_center_x+int(3*w),qr_center_y+int(3*w)), 255, -1)	
			removesticker = cv2.bitwise_and(removesticker, mask)
			removesticker = utility.cnctfill(removesticker)
			qr_img[removesticker == 255] = 0
	else:
		qr_count = 0		
		img_h, img_w, _ = qr_img.shape
		split_width = int(qr_window_size)
		split_height = int(qr_window_size)
	
		X_points = utility.start_points(img_w, split_width, overlap)
		Y_points = utility.start_points(img_h, split_height, overlap)

		for i in Y_points:
			for j in X_points:
				split = qr_img[i:i+split_height, j:j+split_width]
				qr_count += 1
				b,g,r = cv2.split(split)		
				otsu,_ = cv2.threshold(g, 0, 255, cv2.THRESH_OTSU)
				mask = cv2.threshold(g, int(otsu*1.30),256, cv2.THRESH_BINARY)[1]
				thresholded = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)				
				inverted = thresholded

				if debug is True:
					cv2.namedWindow('[DEBUG] [QR] Scanning QRcode', cv2.WINDOW_NORMAL)
					cv2.resizeWindow('[DEBUG] [QR] Scanning QRcode', 1000, 1000)
					cv2.imshow('[DEBUG] [QR] Scanning QRcode', inverted); cv2.waitKey(2000); cv2.destroyAllWindows()

				id = decode(inverted)
				if id != []:
					qr_proof = inverted
					for QRcode in decode(inverted):
						QRcodeRect = QRcode.rect
						QRcodeData = QRcode.data.decode("utf-8")
						QRcodeType = QRcode.type	

					#try to remove sticker 
					(x, y, w, h) = QRcodeRect
					removesticker = np.zeros_like(mask)
					qr_center_y = y+int(w/2)
					qr_center_x = x+int(h/2)
					removesticker = cv2.rectangle(removesticker, (qr_center_y-(2*w),qr_center_y+(2*w)), (qr_center_x-(2*h),qr_center_x+(2*h)), 255, -1)
					removesticker = cv2.bitwise_and(removesticker, mask)
					removesticker = utility.cnctfill(removesticker)
					break
			else:		
				continue
			break
	return QRcodeType, QRcodeData, QRcodeRect, qr_count, qr_proof, removesticker


if __name__ == "__main__":
	print("You are running qr.py solo...")
	
	filename = sys.argv[1]							# Translating arguments into something the function above can understand
	img=cv2.imread(filename)
	
	qr_window_size = sys.argv[2]
	if qr_window_size == "None":
		qr_window_size = None
	else:
		qr_window_size = float(qr_window_size)
	
	overlap = sys.argv[3]
	if overlap == "None":
		overlap = None
	else:
		overlap = float(overlap)

	debug = sys.argv[4]
	if debug == "True":
		debug = True
	else:
		debug = False		
	
	QRcodeType, QRcodeData, QRcodeRect, qr_count, qr_proof, removesticker = qr_scan(img, qr_window_size, overlap, debug) # Run the qr.py module
	
	if QRcodeData == None:
		print("[QR]--{}--No QRcode found".format(filename))	# Log
		QRcodeData = "NA"
	else:
		print("[QR]--{}--Found {}: {} on the {}th iteration".format(filename, QRcodeType, QRcodeData, qr_count))	# Log

	csvname = './initial_qrscan' +'.csv'								# Create CSV and store barcode info
	file_exists = os.path.isfile(csvname)
	with open (csvname, 'a') as csvfile:
		headers = ['Filename', 'QR Code']
		writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n',fieldnames=headers)
		if not file_exists:
			writer.writeheader()
		writer.writerow({'Filename': filename, 'QR Code': QRcodeData})

	print("[QR]--{}--Saved filename and QRcode info to: ./initial_qrscan.csv".format(filename))
