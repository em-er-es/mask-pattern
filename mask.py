#!/usr/bin/env python
## @file
# @title Image processing using masks
# @brief Process input using mask or internal definitions to create output
# @author Ernest Skrzypczyk
# @email emeres.code@onet.eu
# @date 07.2017
# @version 0.8.1
# @licence GPLv3
#

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os.path
import argparse
from matplotlib import rcParams as rcp
rcp.update({'font.size': 10})
# rcp.update({'font.sans-serif': ['Computer Modern Sans serif']})
# rcp.update({'font.serif': ['Computer Modern Roman']})
# rcp.update({'font.family': 'serif'})
rcp.update({'text.usetex': True})

def warn(text):
	print('\033[38;5;226m' + text)

def error(text, errorcode):
	print('\033[38;5;196m' + text)
	exit(errorcode)

#%% Check if run as a script or imported as a library
if __name__ == '__main__':
	parser = argparse.ArgumentParser(prog = os.path.basename(__file__))
	## @todo Add epilog = 'description'
	parser = argparse.ArgumentParser(description = 'Perform basic image processing on input by using a mask to generate output.')
	parser.add_argument('-bt', '--binary-threshold', dest = 'binaryThreshold', type = int, default = 0, help = 'Threshold for binary filtering', metavar = 'BINARYTHRESHOLD <0-255><!1>') #Definition threshold for binary filtering and default value
	parser.add_argument('-i', '--input', dest = 'filenameInput', type = str, default = 'input.png', help = 'Input image') #Option for input image
	parser.add_argument('-im', '--internal-mask', dest = 'internalMask', type = int, default = 0, help = 'Internal mask') #Option for internal mask
	parser.add_argument('-in', '--invert', dest = 'invert', type = int, default = 0, help = 'Invert input, mask, output') #Option for inverting
	parser.add_argument('-g', '--grayscale', dest = 'grayscale', type = int, default = 0, help = 'Convert color space of input, mask, output') #Option for color space conversion
	parser.add_argument('-f', '--filter', dest = 'filtering', type = int, default = 0, help = 'Use the mask as a filter') #Option for filtering
	parser.add_argument('-m', '--mask', dest = 'filenameMask', type = str, default = 'mask.png', help = 'Mask image') #Option for mask image
	parser.add_argument('-o', '--output', dest = 'filenameOutput', type = str, default = 'output.png', help = 'Output image') #Option for output image
	parser.add_argument('-sc', '--separate-mask-channels', dest = 'separateMaskChannels', type = int, default = 0, help = 'Separate mask channels') #Option for mask channel separation
	parser.add_argument('-sm', '--scale-mask', dest = 'scaleMask', type = float, default = 0.0, help = 'Scale mask') #Option for scaling mask
	parser.add_argument('-so', '--select-output', dest = 'outputSelector', type = int, default = 1, help = 'Select output: 0 -- Input; 1 -- !Output; 2 -- Output mask; 3 -- Mask; 4 -- Filter; * -- Output without scaling', metavar = 'OUTPUTSELECTOR <0-b4>') #Option for output selection
	parser.add_argument('-sf', '--scale-to-fit', dest = 'scaleFit', type = float, default = 0.0, help = 'Scale mask to fit') #Option for scaling mask to fit input image dimensions
	parser.add_argument('-t', '--threshold', dest = 'threshold', type = float, default = -85, help = 'Threshold for mask', metavar = 'THRESHOLD <-100-255><!-85>')
	parser.add_argument('-si', '--show-images', dest = 'showImages', type = int, default = 0, help = 'Show images', metavar = 'SHOWIMAGES <0-b4><!0>') #Option for displaying images
	parser.add_argument('-s', '--save-images', dest = 'saveImages', default = True, action = 'store_true', help = 'Save images') #Option for saving processed images
	parser.add_argument('-sp', '--save-path', dest = 'savePath', type = str, default = '/tmp', help = 'Definitions path for generated images, implies save images option', metavar = '<SAVEPATH><!/tmp>') #Definition of save path for processed images
	parser.add_argument('-v', '--verbose', dest = 'verbose', type = int, default = 0, help = 'Set verbose level') #Option for verbose
	args = parser.parse_args() #Parse arguments and file mask from command line

	# Configuration from command line
	binaryThreshold = int(args.binaryThreshold)
	filenameInput = str(args.filenameInput)
	filenameMask = str(args.filenameMask)
	filenameOutput = str(args.filenameOutput)
	filtering = int(args.filtering)
	grayscale = int(args.grayscale)
	internalMask = int(args.internalMask)
	invert = int(args.invert)
	threshold = int(args.threshold)
	separateMaskChannels = int(args.separateMaskChannels)
	saveImages = bool(args.saveImages)
	scaleMask = float(args.scaleMask)
	scaleFit = float(args.scaleFit)
	outputSelector = int(args.outputSelector)
	savePath = str(args.savePath)
	showImages = int(args.showImages)
	verbose = int(args.verbose)

	if savePath != '/tmp': # Save path default value
		saveImages = True
	if filtering > 0:
		outputSelector = 4
	if grayscale < 0:
		grayscale = - grayscale
		separateMaskChannels = 1
	if separateMaskChannels > 0:
		grayscale &= not 0b00000001

	if verbose > 1:
		pass
		# @todo Print all argument settings
		print('Arguments settings')
else:
	error('No implementation of class or library yet', 127)

if os.path.isfile(filenameInput):
	InputGrayscale = cv2.imread(filenameInput, cv2.IMREAD_GRAYSCALE)
	if grayscale & 0b00000001:
		Input = InputGrayscale
	else:
		Input = cv2.imread(filenameInput, cv2.IMREAD_COLOR)
else:
	error('No input file:\t' + filenameInput, 1)

Mask41 = np.array([[0., 0., 0., 0.],
					[1., 1., 1., 1.],
					[1., 1., 1., 1.],
					[0., 0., 0., 0.]],
					dtype = np.uint8)

Mask42 = np.array([[0., 1., 1., 0.],
					[0., 1., 1., 0.],
					[0., 1., 1., 0.],
					[0., 1., 1., 0.]],
					dtype = np.uint8)

Mask51 = np.array([[1., 1., 1., 1., 1.],
					[1., 4., 2., 4., 1.],
					[1., 2., 8., 2., 1.],
					[1., 4., 2., 4., 1.],
					[1., 1., 1., 1., 1.]],
					dtype = np.uint8)

Mask510 = np.array([[-1., -1., -1., -1., -1.],
					[-1., -4., -2., -4., -1.],
					[-1., -2., 8., -2., -1.],
					[-1., -4., -2., -4., -1.],
					[-1., -1., -1., -1., -1.]],
					dtype = np.int8)

Mask52 = np.array([[0., 0., 0., 0., 0.],
					[0., 0., 1., 0., 0.],
					[0., 1., 1., 1., 0.],
					[0., 0., 1., 0., 0.],
					[0., 0., 0., 0., 0.]],
					dtype = np.uint8)

Mask520 = np.array([[0., 0., -1., 0., 0.],
					[0., 0., -1., 0., 0.],
					[-1., -1., 8., -1., -1.],
					[0., 0., -1., 0., 0.],
					[0., 0., -1., 0., 0.]],
					dtype = np.int8)

Mask91 = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0.],
					[0., 0., 0., 0., 0., 0., 0., 0., 0.],
					[0., 0., 0., 0., 1., 0., 0., 0., 0.],
					[0., 0., 0., 1., 1., 1., 0., 0., 0.],
					[0., 0., 1., 1., 1., 1., 1., 0., 0.],
					[0., 0., 0., 1., 1., 1., 0., 0., 0.],
					[0., 0., 0., 0., 1., 0., 0., 0., 0.],
					[0., 0., 0., 0., 0., 0., 0., 0., 0.],
					[0., 0., 0., 0., 0., 0., 0., 0., 0.]],
					dtype = np.uint8)

Mask92 = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0.],
					[0., 1., 1., 1., 1., 1., 1., 1., 0.],
					[0., 1., 1., 1., 1., 1., 1., 1., 0.],
					[0., 1., 1., 0., 0., 0., 1., 1., 0.],
					[0., 1., 1., 0., 0., 0., 1., 1., 0.],
					[0., 1., 1., 0., 0., 0., 1., 1., 0.],
					[0., 1., 1., 1., 1., 1., 1., 1., 0.],
					[0., 1., 1., 1., 1., 1., 1., 1., 0.],
					[0., 0., 0., 0., 0., 0., 0., 0., 0.]],
					dtype = np.uint8)

if int(internalMask) == 0 and os.path.isfile(filenameMask):
	if verbose > 0:
		print('Using mask file: ' + filenameMask)
	if grayscale & 0b00000100:
		Mask = cv2.imread(filenameMask, cv2.IMREAD_GRAYSCALE)
	else:
		Mask = cv2.imread(filenameMask, cv2.IMREAD_COLOR)
elif int(internalMask) != 0:
	# exec('Mask = 255 * Mask' + str(internalMask).zfill(2))
	exec('Mask = Mask' + str(internalMask).zfill(2))
else:
	warn('No input mask file:' + filenameMask + '\nUsing internal mask')
	# Predetermined mask
	# Mask = 255 * Mask41
	Mask = 255 * Mask91

	if len(Mask.shape) < 3:
		rows = []
		for i in range(Mask.shape[0]):
			columns = []
			for j in range(Mask.shape[1]):
				columns.append([Mask[i][j], Mask[i][j], Mask[i][j]])
			rows.append(columns)
		Mask = np.array(rows, dtype = np.uint8)

# Mask dimensions
xm = int(Mask.shape[0])
ym = int(Mask.shape[1])
# Image dimensions
xi = int(Input.shape[0])
yi = int(Input.shape[1])

if xm > xi or ym > yi:
	warn('Mask is larger than input image')

# Binarize mask
if binaryThreshold > 0:
	_, Mask = cv2.threshold(Mask, 1, binaryThreshold, cv2.THRESH_BINARY)

# Scale mask
if scaleMask != 0 and scaleFit == 0:
	Mask = cv2.resize(Mask, dsize = 0, fx = scaleMask, fy = scaleMask, interpolation = cv2.INTER_LINEAR)
elif scaleFit != 0:
	Mask = cv2.resize(Mask, dsize = (xi, yi), interpolation = cv2.INTER_LINEAR)

# Invert
if invert > 0:
	# Decompose inverting option
	# Invert input
	if invert & 0b00000001:
		Input = 255 - Input
	# Invert mask
	if invert & 0b00000100:
		Mask = 255 - Mask

# Apply threshold to mask elements -- initial parameters
if threshold != 0:
	if len(Mask.shape) > 2 and Mask.shape[2] > 1:
		_ = cv2.cvtColor(Mask, cv2.COLOR_RGB2GRAY).flatten()
	else:
		_ = Mask.flatten()
	MaskLength = len(_)
	MaskZero = np.bincount(_)[0]
	MaskPositive = MaskLength - MaskZero
	if threshold < 0:
		_ = np.int(np.floor(- threshold / 100.0 * MaskPositive))
		if verbose > 0:
			print('Using relative threshold: ' + str(- threshold) + '%, ' + str(_) + 'px')
		threshold = _
	elif verbose > 0:
		print('Using absolute threshold: ' + str(threshold) + 'px')

# Print mask
if verbose > 1:
	print(Mask)

# Output initialization
Output = np.zeros((xi, yi, 3), dtype = np.uint8)

# Mask fitted into input
# Along x
m = np.int(np.ceil(xi / xm))
# Along y
n = np.int(np.ceil(yi / ym))

if not grayscale & 0b00000001 or grayscale & 0b00000010:
	z = 3
else:
	z = 1

if grayscale & 0b00000100:
	_ = np.zeros((xm, ym, 3), dtype = np.uint8)
	for i in range(3):
		_[:, :, i] = Mask[:, :]
	Mask = _

# Initialize empty mask
MaskEmpty = np.zeros((xm, ym), np.uint8)
# Initialize composed mask
OutputMask = np.zeros((xi, yi, z), np.uint8)

# Perform less calculations by reusing last cycle computations
b = 0
d = 0

if outputSelector != 0 and outputSelector != 3 and outputSelector != 4:
	#%% Compose output mask
	for k in range(m):
		## @todo Double check if k * xm on all iterations is faster
		if m > 0:
			a = b
		else:
			a = (k + 0) * xm
		b = (k + 1) * xm

		for l in range(n):
			if l > 0:
				c = d
			else:
				c = (l + 0) * ym

			d = (l + 1) * ym
			if verbose > 1:
				print(a, b, c, d)

			for channel in range(z):
				if threshold > 0:
					# Count positive values of applied mask to input image section
					if separateMaskChannels > 0:
						j = MaskLength - np.bincount((Input[a:b, c:d, channel] * Mask[:, :, channel]).flatten())[0]
#						j = MaskLength - np.bincount(np.dot(Input[a:b, c:d, channel], Mask[:, :, channel]).flatten())[0]
					else:
						j = MaskLength - np.bincount((InputGrayscale[a:b, c:d] * Mask[:, :, channel]).flatten())[0]
#						j = MaskLength - np.bincount(np.dot(InputGrayscale[a:b, c:d], Mask[:, :, channel]).flatten())[0]
					# print('j ', j, 'threshold', threshold)
					if j < threshold:
						OutputMask[a:b, c:d, channel] = MaskEmpty
						continue
				OutputMask[a:b, c:d, channel] = Mask[:, :, channel]

	# Copy channels
	if grayscale & 0b00000010:
		OutputMask[:, :, 1] = OutputMask[:, :, 0]
		OutputMask[:, :, 2] = OutputMask[:, :, 0]

	# Invert
	if invert > 0:
		# Invert output mask
		if invert & 0b00001000:
			OutputMask = 255 - OutputMask

MaskOutput = OutputMask

# Use specified output
if outputSelector == 0:
	# Input
	Output = Input
elif outputSelector == 1:
	# Processed file
	amax = np.max(Output)
	# bmax = np.max(Mask)
	bmax = np.max(OutputMask)
	cmax = np.max(Input)
	vmax = np.max([amax, bmax, cmax, 1])
	if grayscale & 0b00000001:
		_ = np.zeros((xi, yi, 1), dtype = np.uint8)
		_[:, :, 0] = Input[:, :]
		Input = _
	OutputMask = OutputMask / vmax
	Output = Input * OutputMask
elif outputSelector == 2:
	# Output mask
	Output = OutputMask
elif outputSelector == 3:
	# Mask
	Output = Mask
elif outputSelector == 4:
	# Filter
	amax = np.max(Input)
	bmax = np.max(Output)
	vmax = np.max([amax, bmax, 1])
	if len(Mask.shape) > 2 and Mask.shape[2] > 1:
		Mask = cv2.cvtColor(Mask, cv2.COLOR_RGB2GRAY)
	Output = cv2.filter2D(Input, -1, Mask)
elif outputSelector == 5:
	# Processed file with filled fields mask
	amax = np.max(Output)
	bmax = np.max(OutputMask)
	cmax = np.max(Input)
	vmax = np.max([amax, bmax, cmax, 1])
	if grayscale & 0b00000001:
		_ = np.zeros((xi, yi, 1), dtype = np.uint8)
		_[:, :, 0] = Input[:, :]
		Input = _
	OutputMask = OutputMask / vmax
	Output = Input * OutputMask
	# Perform less calculations by reusing last cycle computations
	b = 0
	d = 0

	for k in range(m):
		if m > 0:
			a = b
		else:
			a = (k + 0) * xm
		b = (k + 1) * xm

		for l in range(n):
			if l > 0:
				c = d
			else:
				c = (l + 0) * ym

			d = (l + 1) * ym
			if len(Mask.shape) > 2 and Mask.shape[2] > 1:
				Mask = cv2.cvtColor(Mask, cv2.COLOR_RGB2GRAY)
			if len(Output.shape) > 2 and Output.shape[2] > 1:
				for channel in range(3):
					sample = Output[a:b, c:d, channel]
					# sample = Output[a:b, c:d, channel].ravel()[np.flatnonzero(Output[a:b, c:d, channel])]
					average = np.mean(sample, dtype = np.float64)
					vmax = sample.max()
					Output[a:b, c:d, channel] = cv2.multiply(average, Mask) # @todo Why is the vmax parameter < 0.29 not working
					# Output[a:b, c:d, channel] = average * (Mask / (0.5 * vmax)) # @todo Why is the vmax parameter < 0.29 not working
					# Output[a:b, c:d, channel] = average * (Mask / (1 * vmax)) # @todo Why is the vmax parameter < 0.29 not working
					# Output[a:b, c:d, channel] = average * (Mask / (0.3 * vmax)) # @todo Why is the vmax parameter < 0.29 not working
					# Output[a:b, c:d, channel] = average * Mask
			else:
				Output[a:b, c:d, channel] = Output[a:b, c:d, channel].nonzero().mean() * Mask
else:
	# Processed file without scaling
	Output = Input * OutputMask

# Change color space to gray scale
if grayscale > 0:
	if (grayscale & 0b00000100 or grayscale & 0b00001000) and not grayscale & 0b00000001:
		Output = cv2.cvtColor(Output, cv2.COLOR_RGB2GRAY)
	if grayscale & 0b00001000 and not (grayscale & 0b00000001 or grayscale & 0b00000010):
		Output = cv2.cvtColor(Output, cv2.COLOR_GRAY2RGB)

# Invert output
if invert & 0b00000010:
	Output = 255 - Output

if saveImages:
	if grayscale & 0b00000001 and not grayscale & 0b00000100 and len(Output.shape) > 2 and Output.shape[2] > 1:
		_ = np.zeros((xi, yi), dtype = np.uint8)
		_[:, :] = Output[:, :, 0]
		Output = _
		Output = cv2.cvtColor(Output, cv2.COLOR_GRAY2RGB)
	if grayscale & 0b00000010 and not grayscale & 0b00000100 and not grayscale & 0b00001000 and len(Output.shape) > 2 and Output.shape[2] > 1:
		Output = cv2.cvtColor(Output, cv2.COLOR_BGR2GRAY)
	cv2.imwrite(filenameOutput, Output)

if showImages > 0:
	if len(Input.shape) > 2 and Input.shape[2] > 1:
		Input = cv2.cvtColor(Input, cv2.COLOR_BGR2RGB)
	else:
		Input = cv2.cvtColor(Input, cv2.COLOR_GRAY2RGB)
	plt.subplot(141)
	plt.imshow(Input)
	# plt.title('Original')
	plt.title('Input')
	plt.xticks([])
	plt.yticks([])

	if len(Mask.shape) > 2 and Mask.shape[2] > 1:
		Mask = cv2.cvtColor(Mask, cv2.COLOR_BGR2RGB)
	else:
		Mask = cv2.cvtColor(Mask, cv2.COLOR_GRAY2RGB)
	plt.subplot(142)
	plt.imshow(Mask)
	# plt.title('Mask')
	# plt.title('Mask ' + str(Mask.shape[0]) + 'x' + str(Mask.shape[1]) + 'px')
	plt.title(r'''\centering Mask \newline \small{''' + str(Mask.shape[0]) + 'x' + str(Mask.shape[1]) + '''}''')
	# plt.title(r'''\centering Mask \newline \small{''' + str(Mask.shape[0]) + 'x' + str(Mask.shape[1]) + '''px}''')
	plt.xticks([])
	plt.yticks([])

	# if len(OutputMask.shape) > 2 and OutputMask.shape[2] > 1:
	# 	OutputMask = cv2.cvtColor(OutputMask, cv2.COLOR_BGR2RGB)
	# else:
	# 	OutputMask = cv2.cvtColor(OutputMask, cv2.COLOR_GRAY2RGB)
	plt.subplot(143)
	plt.imshow(MaskOutput)
	# plt.imshow(OutputMask)
	# plt.title('Averaging')
	plt.title('Output mask')
	plt.xticks([])
	plt.yticks([])

	if len(Output.shape) > 2 and Output.shape[2] > 1:
		Output = cv2.cvtColor(Output, cv2.COLOR_BGR2RGB)
	else:
		Output = cv2.cvtColor(Output, cv2.COLOR_GRAY2RGB)
	plt.subplot(144)
	plt.imshow(Output)
	# plt.title('Averaging')
	plt.title('Output')
	plt.xticks([])
	plt.yticks([])

	plt.show()
