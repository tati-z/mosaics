# Tatiana Zihindula
# C16339923
# "Group" Project
# 23/11/2019

# This module contains utility functions that are used throughout the
# creation of mosaic filters.

import cv2
import numpy as np
import random, math

def add_edges(pts, width, height):
	'''
	Adds the edges coordinates of the screen into the points array
	
	when edges are not added, the image using both triangles or pentagon tiles
	could have the corners non texturised.
 
	[INFINITY BUG FIXED]
	The offset is used to bring infinity regions outside of the screen in the vonroi tiles
	and prevents round edges for triangular tiles.

	reference: https://stackoverflow.com/questions/20515554/colorize-voronoi-diagram
	
	'''
	o = 512 # the offset to bring infinite points off the screen
	pts = np.append(pts, [[-o,-o], [width+o, -o], [-o, height+o], [width+o, height+o]], axis=0)
	return pts

def map_range(value, from_0, to_0, from_1, to_1):
	'''
	Maps a value from one range to another.

	This is equivalent to the changing of basis in linear algebra.

	Reference: https://www.youtube.com/watch?v=P2LTAUO1TdA

	i.e. if i have range1 = 0 - 255 and range2 10 - 0
	what would a value x in range 1 map to value y in range 2?
	looking at range1 and range2, and given value=x
	if x = 0 in range 1   -> x will map to 10 in range 2
	if x = 255 in range 1 -> x will map to 0 in range 2
	if x is  somehwere in betwerrn in range1, x will map somewhere in between in range2

	e.g.

	>>> map_range(0, 0, 255, 10, 0)
	10.0
	>>> map_range(255, 0, 255, 10, 0)
	0.0
	>>> map_range(127, 0, 255, 10, 0)
	5.019607843137255

	This method is particulary useful in the fibonacci_mesh where I wanted a grayscale value
	closer to 0 to map to a higher index in the fibonacci array, and a higher value close
	to 255 to map to small indexes in the fibocacci array
	'''
	return from_1 + (to_1 - from_1) * ((value - from_0) / (to_0 - from_0))

def randrange_pts_2d(x1, x2, y1, y2, count):
	'''
	Returns 'count' random pairs of indexes from a 2D array
	x1-x2 the range of the width index
	y1-y2 the range of the width index
	e.g.
	>>> import numpy as np
	>>> a = np.arange(28)
	>>> a.resize((7,4))
	>>> a
	array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15],
           [16, 17, 18, 19],
           [20, 21, 22, 23],
           [24, 25, 26, 27]])
	>>> randrange_pts_2d(1,3,4,6,3)
	array([[0, 0],
          [2, 4],
          [3, 5],
          [2, 4]])
	
	As shown, apart from the [0,0] added in the initialisation, there are 3 random indexes
	from the range provided. 
	These indexes can now be used to retrieve the value from the aaray
	'''
	points = np.zeros((1,2), dtype=int)
	for c in range(count):
		i = random.randint(x1,x2)
		j = random.randint(y1,y2)
		points = np.append(points,[[i,j]], axis=0)
	return points

def normalise(vertices, img):
	'''
	Normalise vertices from [0,1] to [0,width), [0,height) if the maximum value of the vertices
	is < 1. else ranges are already overlapping, so no need to normalise.
	'''
	v = vertices
	if np.max(vertices) < 1:
		height, width = img.shape[:2]
		# normalise vertises x from range [0,1[ to [0,width[
		v[:,0] *= width
		# normalise vertises y from range [0,1[ to [0,height[
		v[:,1] *= height
		# get rid of the floating point from previous range [0,1[
	return np.int32(v)

def fill(img, vertices, contour=None, isblank=False):
	'''
	Fills the region inide the images with polygons whose coordinates are in the vertices array.
	'''
	for v in vertices:
		if len(v) < 1: # empty vertices are of no use, maybe 2, too, but might be a contour line. 
			continue
		if isblank: # speed up. just the contour overlay. no fill needed.
			contour = 0 if contour is None else contour
			img = cv2.polylines(img, [v], True, contour, 1)
			continue
		# normalise the vertices if they are not in the range of the images's coordinates
		v = normalise(v, img)
		# the mask on the image that will isolate the current triangle
		mask = np.zeros(img.shape[:2], dtype=np.uint8)
		# covering the image, but the current
		# this is needed to calculate the average color of this portion.
		mask = cv2.fillPoly(mask, [v], 255)
		# get the average color of the portion covered by the mask
		average = np.array(cv2.mean(img, mask))
		# now fill that portion with the average color computed above.
		cv2.fillPoly(img, [v], average)
		# separate the polygons with a darker version of this average color
		# or the contour color if specified
		if contour is None:
			img = cv2.polylines(img, [v], True, average-16, 1)
		else:
			img = cv2.polylines(img, [v], True, contour, 1)	
	return img


def rotate(xo,yo,x,y,angle):
	"""
	Rotate a point clockwise from an origin other than (0,0)
	Arguments:
	xo: the origin x
	yo: the origin y
	y : the point to rotate's x
	y : the y point to rotate

	angle: an angle in degrees. e.g 45
	This is used for circular mosaics
	"""
	# convert the angle to radians, -1 for clockwise
	angle = (math.pi * angle)/180
	# use the geometrical formulae to rotate a point
	
	px = x + math.cos(angle) * (xo - x) - math.sin(angle) * (yo - y)
	py = y + math.sin(angle) * (xo - x) + math.cos(angle) * (yo - y)
	return int(px), int(py)




def show(*args, **kwargs):
	'''
	Shows one image, or concatenates many images and displays them to the screen
	'''
	titles = kwargs.get('titles') if kwargs.get('titles') else 'images'
	try:
		iter(args)
	except TypeError:
		# only one image was passed
		cv2.imshow(titles, args)
	else:
		# many images were passed: concatenate them
		cv2.imshow(titles, cv2.hconcat(args))
	key = cv2.waitKey(0)

def show_fewer(img, mosaics, gradients, l=0, h=-1):
	'''
	ensures that no more than 3 images are shon at a time.
	when concatenating images, they look small. this function prevents that by only showing max 3 side by side
	'''
	if l > h:
		return
	elif h - l >= 0 and h-l <=2:
		ims = [i[1] for i in mosaics[l:h]]
		labels =['Original'] + [i[0] for i in mosaics[l:h]]
		show(img, *ims, titles = (' '*30).join(labels))
		
		grads = [i[1] for i in gradients[l:h]]
		glabels = ['Original']+ [i[0] for i in gradients[l:h]]
		show(img, *grads, titles = (' '*30).join(glabels))

		cv2.destroyAllWindows()
	else:
		show_fewer(img, mosaics, gradients, l=l, h= (h+l)//2)
		show_fewer(img, mosaics, gradients, l=(h+l)//2, h=h)
