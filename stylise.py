# Tatiana Zihindula
# C16339923
# "Group" Project
# 23/11/2019

# This module contains styling used to manipulate the colors inside the generated mosaics
# ideally I'd want to have an UI to generate your own gradients, but this is due tomorow
# and it hasn't been much of a group project if I am to be honnest.

# stylise would contain the following components
# - Qt UI implemented UI to change the arguments of functions.
# - Mesh editing, by drawing them to the screen, or editing a mesh to remove/add points
# - custom color mapping export features for the end user.
# but//
# 
# anyway, moving on

# TODO: even after a UI gets implemented. maybe have a random option where the program will generate a random combination of styling.. instead of having the user trying them all...
# You never know. unless you tried them all//

import cv2, random, glob

# The separated text effect
#  b.append(' '.join(video.xml_captions[0].text).split('\n'))

# applies a random gradien map to an image 
def map_gradient(img, queue = None):
	'''
	Gradient map maps replaces the colors inside the image by those of the gradient.
	
	but test images run through all of them, so all of them have equal change of being chosen.
    '''
	# todo: create costom gradients
	# more on gradient map: https://docs.opencv.org/2.4/modules/contrib/doc/facerec/colormaps.html
	gradients = {
		"AUTUMN": cv2.COLORMAP_AUTUMN,
		"BONE": cv2.COLORMAP_BONE,
		"JET": cv2.COLORMAP_JET,
		"WINTER": cv2.COLORMAP_WINTER,
		"RAINBOW": cv2.COLORMAP_RAINBOW,
		"OCEAN": cv2.COLORMAP_OCEAN,
		"SUMMER": cv2.COLORMAP_SUMMER,
		"SPRING": cv2.COLORMAP_SPRING,
		"COOL": cv2.COLORMAP_COOL,
		"HSV": cv2.COLORMAP_HSV,
		"PINK": cv2.COLORMAP_PINK,
		"HOT": cv2.COLORMAP_HOT
	}
	the_chosen_one = random.choice(list(gradients.keys()))
	# currently returning a random gradient, for a
	# lack of UI to select and or customise one.
	img = cv2.applyColorMap(img,gradients[the_chosen_one])
	if queue:
		queue.put((the_chosen_one,img)) # the first argument will be used as label for display
	else:
		return img

def gradient_blend(im):
	'''
	Applies a blend of one image into a another, blends well when one is a mask
	because I am using binary operations.

	This will give a better visual effect when used with gradient images.
	I guess waves could also look cool.. fires are dramatic lol..
	'''
	img = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
	imgs = glob.glob('./gradients/*[.jpg, .png]')
	if len(imgs) == 0:
		return "", img
	random.shuffle(imgs)
	
	blends = {"AND": cv2.bitwise_and,
			  "OR":cv2.bitwise_or,
			  "XOR": cv2.bitwise_xor,
			  "NOPE": lambda x, y: x, # every now and then do nothing
	}
	# apply a random blend
	the_chosen_one = random.choice(list(blends.keys()))
	if the_chosen_one == "NOPE":
		return "NO FILTER", img
	
	height, width = img.shape[:2]
	# TODO:FEATURE: just resize the image to make it fit instead of looping no?
	for f in imgs:
		gradient = cv2.imread(f)
		try:
			# if the size of the gradient fits, use it, else, thank you, next.
			gradient = gradient[0:height, 0:width]
			blended  = blends[the_chosen_one](img,gradient)
			return the_chosen_one + ':' + f , blended
		except:
			# go to the next image, this one didn't work
			pass
	# didn't find a size that fits
	return "", img
