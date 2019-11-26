# Tatiana Zihindula
# C16339923
# "Group" Project
# 23/11/2019

# This package contains the implementation of actual currently supported tiles types
# which are: square, triangles and pentagons

# 1. SQUARE TILES

# The square tiles has no variation of shapes, and used no particular heuristics to
# choose the best tiles. it simply subdivises the images into n squares and fills each
# subsquare with the average color in that square.

# 2. TRIANGULAR TILES

# This tile style used the Delaunay triangulation to subdivide the image into tiangles using
# a set of points. more about it here: https://tinyurl.com/t7z9hlh
# 
# The variation of the quality of tiles relies on the partitions of these points within the image
#
# Two partition algorithms are used:
# 1. Random partition
#
# Here, N random idexes are chosed at random from the image.
# The result of this is a somewhat evenly partitioned image, without notion of proportions.
# This partition works particularly well for unstructured, and unproportional images
# e.g. a field, a table, et.c
# see results here: https://tinyurl.com/rreg8ag
# if you looked at the result, you might notice that this partition does not work
# very well on people's faces, not on symetrical images, like fractal flowers.
# due to the fact that it devides everything evenly.
#
# 2. Finonacci Partition (Fibonacci Mesh)
# 
# The fibocacci uses the first 10 digits-ish of the fibonacci sequence to partition the tiles.
# One of the amazing proprety of this sequence is its exhibition of the Golden Ration
# https://www.livescience.com/37704-phi-golden-ratio.htm
#
# Unlike the random partition, the fibocacci partition complements everything with proportions
# and symetry: e.g. People's faces, natural objects, fractal flowers, etc..
# see result here:

# 3. PENTAGONAL TILES

# Pentagonal tiles, or Honey Comb Tiles, or Cell tiles, as I have been thinking to rename them,
# Are created using the the Voronoi Diagrams. discussed in details here: https://tinyurl.com/t7z9hlh
#
# In addition to the Random Partitioning and the Fibonacci Partitioning from the Triangular tiles
# pentagonal tiles adds another partitioning called the Lloyd Mesh.
#
# * Lloyd Mesh
# This partitioning is based on the random partitioning combined with the Lloyd algorithm.
# after N itterations the previously randomly choosen get rearanged in a more uniform way.
# This was implemented using the Kmean clustering in openCV which uses a similar implementation.
# more on that here: https://en.wikipedia.org/wiki/Lloyd%27s_algorithm
# more on that here: https://en.wikipedia.org/wiki/K-means_clustering
#
# this partitioning only looks better when the output needs to be evenly divided.
# for proportional images, its output looked more flat even compared to the random partition.

# 4. CIRCULAR TILES

# Inspired by halftoned dots, circlar tiles use the intensity within the grascale version of the image
# to decide both the radius to use for that convolution, and the fill color (value of the grayscal)
# color to use for the mask.

# TODO: use the factory design pattern in here for code reuse. also add this to the slies as improvements
# TODO: compress the image before working with it.. but reconstruct it back. this will also help gradients images take less time..
# see https://www.researchgate.net/publication/328190353_IMAGE_COMPRESSION_USING_HAAR_WAVELET_TRANSFORM/link/5bbd99a192851c7fde376351/download

import cv2
import random
import numpy as np
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d, delaunay_plot_2d

# local import
from mesh import lloyd_mesh, fib_mesh, random_pts
from util import fill, map_range, rotate
from stylise import gradient_blend

def pentagons(im, points=[], contour=None, fib_step=None, lloyd_cells=None, isblank=False, queue=None):
	'''
	Divides the image into hexagon/pentagon regions
	Arguments:

	im: the image to work with.

	points: a set of prefered points to use instead of provided option. Usually cached points
	from previous itterations.

	countour: the color of the contour to use. By default the contour is set to 30 intensity
	darker than the color in the current region for contrast.
	
	mesh: when True, a finonacci mesh will be used, but only if points is []

	lloyd: when True, a lloyd mesh will be used, but only if points is []

	isblank: when True, the image will just have the contour of the mesh overlayed when
	filled. This flag treats the image as blank. if contour is set, the overlay will have the
	contour=contour, else, 0 (black)

	queue: when not None, the image is sent to the message queue instead of being returned.
	This is particulal handy when running concurently.
	'''
	# copy the image, to prevent inplace operations
	img = im.copy()
	
	if len(points) != 0:
		# we got free points to work with
		pass
	elif fib_step:
		# use fibonacci mesh
		points = fib_mesh(img, fib_step)
	elif lloyd_cells:
		# use lloyd mesh
		points = lloyd_mesh(img, lloyd_cells)
	else:
		# use random points
		points = random_pts(img)
		
	# create the voronoi dragrams
	vor = Voronoi(points)
	
	# get all the vertices ready and sent them to be filled or contoured
	v = []
	# get all the vertices of this region and
	# check if the current region isn't reaching towards infinity
	for region in vor.regions:
		if not -1 in region and len(region) > 0:
			v.append(vor.vertices[region])
	# fill all regions at once
	fill(img, v, contour=contour, isblank=isblank)

	# if running concurently send the items to the queue
	if queue:
		label = 'Pentagons: '
		label += 'Fibonacci-Steps: {} '.format(fib_step) if fib_step is not None else ''
		label += 'Lloyd-Cell-Size: {} '.format(lloyd_cells) if lloyd_cells is not None else ''
		label += 'Random-Steps: {} '.format(True) if fib_step is None and lloyd_cells is None else ''
		queue.put((label, img))
	else:
		# return them instead.
		return img

	
def triangles(im, points=[], contour=None, fib_step = None, isblank=False, queue=None):
	'''
	Divides the image into triangular regions.

	Arguments:

	im: the image to work with.

	points: a set of prefered points to use instead of provided option. Usually cached points
	from previous itterations.

	countour: the color of the contour to use. By default the contour is set to 30 intensity
	darker than the color in the current region for contrast.
	
	mesh: if not none, denotes the chunck size that the finonacci mesh will be used, but only if points is []

	lloyd: when True, a lloyd mesh will be used, but only if points is []

	isblank: when True, the image will just have the contour of the mesh overlayed when
	filled. This flag treats the image as blank. if contour is set, the overlay will have the
	contour=contour, else, 0 (black)

	queue: when not None, the image is sent to the message queue instead of being returned.
	This is particulal handy when running concurently.
	'''
	# copy the image, to prevent inplace operations
	img = im.copy()
	
	if len(points) != 0:
		# we got free points to work with
		pass
	elif fib_step:
		# use fibonacci mesh
		points = fib_mesh(img, fib_step)
	else:
		# use random points
		points = random_pts(img)
		
	# create Delaunay triangles
	tri = Delaunay(points)
	# get all the vertices of the triangles to sent them to be filled
	v = [points[x] for x in tri.simplices]

	# fill all the vertices.
	fill(img, v, contour=contour, isblank=isblank)
	# if running concurently send the items to the queue
	if queue:
		label = 'Triangles: '
		label += 'Fibonacci-Steps: {} '.format(fib_step) if fib_step is not None else ''
		label += 'Random-Steps: {} '.format(True) if fib_step is None else ''
		queue.put((label, img))
	else:
		# return return the image instead.
		return img
	

def squares(im, s=2, queue=None):
	'''
	Subdivises the image into square tiles, of s% of the size of the image each.

	Arguments:
	im: the image to work with.

	s: the % of the image to be used as the side lengh of each square.
	e.g. if the image is 100*100, and s = 2 each square tile will have the size:
	max(width,height) * 2 // 100 -> wich is 2.
	But when the image is 1920 * 1028, and s is 2,
	s will scale well, using 2% of 1920 = 38, as opposed of staying 2. which would look very small.

	queue: when not None, the image is sent to the message queue instead of being returned.
	This is particulal handy when running concurently.
	'''
	s = s if s > 0 and s < 50 else 2
	img = im.copy()
	#  size in percentage
	height, width = img.shape[:2]
	s = (s * max(width, height))// 100
	
	i, j = 0,0
	while i < height:
		j =0
		while j < width:
			y1, y2, x1, x2= i, i+s, j, j+s
			color = img[y1:y2, x1: x2].mean(axis=0).mean(axis=0)
			img[y1:y2, x1:x2] = color
			img = cv2.rectangle(img,(x1,y1),(x2,y2),color-30,thickness=1)
			j += s
		i += s
	if queue:
		label = 'Squares: Tile-Size: {}'.format(s)
		queue.put((label, img))
	else:
		# return return the image instead.
		return img



def circles(im, s = .6, angle = 0, queue=None):
	'''
	Creates circular tiles of the image.
	inspired by: https://www.gettyimages.ie/detail/illustration/monochrome-halftone-dots-wavy-pattern-royalty-free-illustration/626917876

	BUG: The rotation isn't working as intended, on the real image. 
	
	Arguments:

	im: the image to work with
	s: the % of spacing between the tiles.
	
	'''
	
	s = s if s > 0 and s < 50 else 2
	img = im.copy()
	# as working only with the grascale value
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# add the lightness to the image to increase contrast
	img = cv2.equalizeHist(img)
	# set the maximum radius  size in % to keep the proportions
	height, width = img.shape[:2]
	
	s = int(map_range(s, 0, 50, 0, max(width, height)))
	
	canvas = 255 * np.ones_like(img)
	
	for i in range(0, height, s):
		for j in range(0, width, s):
			# go s steps at a time and get the average color in each region.
			y1, y2, x1, x2 = i, i+s, j,  j+s
			# maybe rotate 45 degree here then calculat the mean at 45 degree?
			avg = img[y1:y2, x1: x2].mean(axis=0).mean(axis=0)
			# max radius is 20, map the range of the radius using the avg value above.
			# values closer to white will have small radius
			radius = int(map_range(avg, 0, 255, s//2, 2))
			# The fill color will not be too black, for asthetic reasons. map it as well
			# maximum black = 50, maximum white = 240 (
			# this changes crops the histogram from [0,255] to [50, 240])
			fill_color = int(map_range(avg, 0,255, 5, 200))
			# the center of the circle
			x, y = (x1 + x2)//2 , (y2+y1)//2
			# if the mat has to be rotated
			if angle % 360 != 0:
				# now rotate the coordinates by this value
				x, y = rotate(x-s, y, x, y, 135)
			# draw the circles at this place
			cv2.circle(canvas, (x,y), radius, fill_color, -1)
	canvas = cv2.addWeighted(canvas, .9, img, .1, 2)
	blend_used, canvas = gradient_blend(canvas)
	
	# if running concurently send the items to the queue
	if queue:
		label = 'Circles: '
		label += 'Steps: {} '.format(s)
		label += 'Angle: {} '.format(angle)
		label += 'Blend: {} '.format(blend_used) if blend_used is not '' else ''
		queue.put((label, canvas))
	else:
		# return return the image instead.
		return canvas
