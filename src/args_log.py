import argparse
import logging
import sys
import traceback
import os
import re
import string

def options():
	parser = argparse.ArgumentParser(description="Full pipeline for automted maize ear phenotyping")

	#Required main input
	parser.add_argument("-i", "--image",  help="Path to input image file", required=True)

	#Optional main arguments
	parser.add_argument("-o", "--outdir", help="Provide directory to saves proofs, logfile, and output CSVs. Default: Will save in current directory if not provided.")
	parser.add_argument("-ns", "--no_save", default=False, action='store_true', help="Default saves proofs and output CSVs. Raise flag to stop saving.")
	parser.add_argument("-np", "--no_proof", default=False, action='store_true', help="Default prints proofs on screen. Raise flag to stop printing proofs.")
	parser.add_argument("-D", "--debug", default=False, action='store_true', help="Raise flag to print intermediate images throughout analysis. Useful for troubleshooting.")	

	#QR code options
	parser.add_argument("-qr", "--qrcode", default=False, action='store_true', help="Raise flag to scan entire image for QR code.")	
	parser.add_argument("-r", "--rename", default=True, action='store_false', help="Default renames images with found QRcode. Raise flag to stop renaming images with found QRcode.")	
	parser.add_argument("-qr_scan", "--qr_window_size_overlap", metavar=("[Window size of x pixels by x pixels]", "[Amount of overlap (0 < x < 1)]"), nargs=2, type=float, help="Provide the size of window to scan through image for QR code and the amount of overlap between sections(0 < x < 1).")

	#Color Checker options
	parser.add_argument("-clr", "--color_checker", default="None", help="Path to input image file with refference color checker. If none provided will use default values.", nargs='?', const='', required=False)
	
	#Pixels Per Metric options
	parser.add_argument("-ppm", "--pixelspermetric", metavar=("[Refference length], [in/cm]"), nargs=2, help="Calculate pixels per metric using either a color checker or the largest uniform color square. Provide refference length in 'in' or 'cm'.")
	
	#Find Ears options
	parser.add_argument("-thresh", "--threshold", metavar=("[channel]", "[intensity threshold]", "[invert]"), help="Manual ear segmentation module. Use if K fails", nargs=3, required=False)
	parser.add_argument("-size", "--ear_size", metavar=("[Min area as percent of total image area]", "[Max Area as percent of total image area]"), nargs=2, type=float, help="Ear size filter default: Min Area: 1.5 percent Max Area: 15 percent. Flag with two arguments to customize size filter.")
	parser.add_argument("-filter", "--ear_filter", metavar=("[Max Aspect Ratio]", "[Max Solidity]"), nargs=2, type=float, help="Ear segmentation filter. Default: Max Aspect Ratio: x < 0.6,  Max Solidity: 0.983. Flag with two arguments to customize ear filter.")

	parser.add_argument("-clnup", "--ear_cleanup", default="None", help="Ear clean-up module. Raise flag to turn on with default settings or provide two arguments: Max Area Coefficient of Variation threshold and Max number of iterations to customize ear clean up module.", nargs='?', const='', required=False)
	parser.add_argument("-slk", "--silk_cleanup", default="None", help="Silk decontamination module. Raise flag to turn on with default settings or provide two arguments: Min change in covexity and Max number of iterations to customize silk clean up module.", nargs='?', const='', required=False)
	
	parser.add_argument("-rot", "--rotation", default=True, action='store_false', help="Raise flag to stop ears from roating.")	

	#Cob and shank segmentation options
	parser.add_argument("-t", "--tip", nargs='*', help="Tip segmentation module. Usage: '-t': automatic thresholding, and '-t # # # #' for custom segmentation. Flag with four arguments to customize tip segmentation module with the following parameters: Hue/Sat Channel, Thresholding intensity, percent, Dialate.")
	parser.add_argument("-b", "--bottom", nargs='*', help="Bottom segmentation module. Usage: '-b': automatic thresholding, and '-b # # # #' for custom segmentation. Flag with four arguments to customize bottom segmentation module with the following parameters: Hue/Sat Channel, Thresholding intensity, percent, Dialate.")

	#KRN call
	parser.add_argument("-krn", "--kernel_row_number", default=False, action='store_true', help="Raise flag to call KRN peaks for fresh hybrid ears.")	

	#Hyb call
	parser.add_argument("-grade", "--usda_grade", default=False, action='store_true', help="Raise flag to predict USDA Grade peaks for fresh hybrid ears.")	

	args = parser.parse_args()
	return args

def get_logger(logger_name):
	args = options()
	
	if args.outdir is not None:
		out = args.outdir
	else:
		out = "./"

	console_handler = logging.StreamHandler(sys.stdout)
	console_handler.setFormatter(logging.Formatter("%(asctime)s — %(levelname)s — %(message)s"))
	logger = logging.getLogger(logger_name)
	logger.setLevel(logging.DEBUG) # better to have too much log than not enough
	logger.addHandler(console_handler)
	
	destin = "{}".format(out)
	if not os.path.exists(destin):
		try:
			os.mkdir(destin)
		except OSError as e:
			if e.errno != errno.EEXIST:
				raise
		LOG_FILE = ("{}EarCV.log".format(out))
	else:
		LOG_FILE = ("{}EarCV.log".format(out))
		
	file_handler = logging.FileHandler(LOG_FILE)
	
	file_handler.setFormatter(logging.Formatter("%(asctime)s — %(levelname)s — %(message)s"))
	
	logger.addHandler(file_handler)
	# with this pattern, it's rarely necessary to propagate the error up to parent
	logger.propagate = False
	return logger