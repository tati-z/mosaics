# Tatiana Zihindula
# C16339923
# "Group" Project
# 23/11/2019

# This mudule includes the implentation of various mesh types used to create mosaic shapes.
import cv2
import numpy as np
import matplotlib.pyplot as plt

from util import add_edges, randrange_pts_2d, map_range, normalise

def fib_mesh(im, step):
	'''
	Returns an array of points computed with the golden ration.

	This algoritm uses the fibocacci sequence, it does not recalculate it. it just uses it.
	
	Arguments:

	im: the image to work with.
	
	step: used to define the cell size to use. must be > 0 and <= 50.
	but beware < 10 probably too small for high res.
	100% means skip 100% percent of the image at a time. not useful, ain't it?!
	'''
	# check if not backward steps and/or more than 50%
	assert step > 0 and step <= 50

	# copy the image to prevent inplace operations
	img = im.copy()
	height, width = img.shape[:2]
	
	# enter the gray area
	if len(img.shape) > 2:
		img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	
	# convert the step into percentage to prevent uneven ratios based on resolution
	s = (step * width)// 100

	# init an array of points, dunno why numpy always gives you an element at array init
	# just override it when I start writting to it damn it!!
	points = np.zeros((1,2), dtype=int)
	
	# using this slightly modified fibonacci sequence. dense areas with opacity ~ 0
	# will be assigned to larger index number in the fibonacci_ish array
	# (creating smaller triangles) whist
	# brighter points will have lesser points (creating) larger triangles.
	# more points (max = 144 for 0) compared to the light areas (min = 1 for 255)

	# I modified this sequence a bit by removing the first 2 digits to have
	# a more evenly distributed version for my unevely distributed needs.
	# 
	fibonacci_ish = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
	# Todo: divide the image in CHUNK sizes and send CHUNKS to each of N treads to compute
	# these points concurently to speed up things.
	# CHUNK_SIZE = 100 # any image larger than 100 * 100 will be divided into chuncks of 100 * 100
	# from threading import Thread
	# from queue import Queue
	# from multiprocessing import lock (one thread at a time can append their chunck points)
	# q = Queue()
	# def work_a_chunk(w, h, q):
	# 	pass

	# or be lazy: to keep things simple!
	for i in range(0, height, s):
		for j in range(0, width, s):
			# go s steps at a time and get the average color in each region.
			y1, y2, x1, x2 = i, i+s, j,  j+s
			avg = img[y1:y2, x1: x2].mean(axis=0).mean(axis=0)
			# map this average color to the number of points we need to draw in this region
			# if the average is too dark, the number will be closer to len(fib-ish) = will
			# results in many random points being choosen there. as fib numbers go high
			# depending on the index of the fib number.
			num_pts = int(map_range(avg, 0,255, len(fibonacci_ish)-1, 0))
			# now get fib_is[num_points] points from this region.
			new_points = randrange_pts_2d(x1,x2,y1, y2, fibonacci_ish[num_pts])
			# append these points to the array.
			# REMINDER: this could have been the critical region if used concurently.
			points = np.append(points,new_points, axis=0)

	# add the edges to prevent clipping. and we are done!
	return add_edges(points, *im.shape[:2])

# allow to make the points denser. 
def random_pts(im, edges = True):
	'''
	Returns an array of random points coordinates withing the image's bounds.
	Arguments:

	im: the image from which bound the points should conform.
	
	edges: when False, edges are not added to the points set.
	'''
	# get the shape of the image
	height, width = im.shape[:2]
	# if the width is larger use it as the upper bound, else use the min
	# coordinates are represented as (x,y); thus 2.
	points = np.random.rand(max(width,height), 2)
	# normalise before adding the borders as the borders use a larger range
	# than np.random.rand which foes from [0..1)
	points = normalise(points, im)

	if edges:
		# now add the edges to the random points list
		points = add_edges(points, width, height)
	return points



def lloyd_mesh(im, cells=10, itter=10):
	'''
	Returns an array of random points coordinates whose shapes are somewhat uniform.

	This algoritm implements Kmean clustering.
	Arguments:

	im: the image to work with.

	cells: the % of cell count to have one the screen. 100 means .75% of the image, as 100% will be completly full
	
	itter: the number of itteration used to cluster regions.
	'''
	assert cells >= 1 and cells <= 100, 'tiles cannot be larger than 100% of the image'
	assert itter > 0 and itter < 50, 'invalid iteration number (jic someone puts infinity)'

	# copy the image, to prevent inplace operations
	img = im.copy()

	# the max of either width or height will decide the ratio
	cells = int(map_range(cells, 1, 100, max(*im.shape[:2]), 1))

	# get initial random points. Do not include edges as Kmean will disrupt them.
	points = random_pts(im, edges=False)
	# create the clusters
	# Define criteria = ( type, max_iter = 10 by default, epsilon = 1.0 )
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, itter, 1.0)
	flags = cv2.KMEANS_RANDOM_CENTERS
	# Apply KMeans
	_,labels,centers = cv2.kmeans(np.float32(points),cells,None,criteria,itter,flags)
	# for i in range(cells):
		# plt.scatter(points[labels.ravel() == i][:,0],points[labels.ravel() == i][:,1])
	# optionally plot it and see how many points are now only grouped into n regions
	# plot.show()
	print('--add done: sent for display---')
	return add_edges(centers, *im.shape[:2])


# TODO: would it be necessary to replace n random points around even if we cached it already?
# caching points makes the mesh predictable? replacing M points with other lerped M points would prevent stalenes?
class MeshCache:
	'''
	The mesh is only computed once per image. this allows to speed things up.
	
	a mesh can take longer to run, depending on the size of the image.
	so only compute it once per image/process and have it chashed and shared.

	e.g. if pentagon tiles of img 1, needs a fibonacci mesh, or args x,y. this same
	mesh was computed earlier. so give that back. else compute it and cache it.
	for the next process if needed.

	same for lloyd mesh.
	'''
	_fib_mesh = {}
	_lloyd_mesh = {}
	
	def __init__(self, img):
		# only allow one cache per image.
		self.img = img
	
	def fib_cache(self, step, lock, memoise=False, max_ = 20):
		'''
		Compute the fib mesh of size step, for current image and cashes it
		because the fib sequence, i can use memoisation here especially for tests. 
		cache for fib_step 28 should be cache of fib 29 + 30. just add the points together and return them to me
		'''
		if self._fib_mesh.get(step) is not None:
			print('~~~ reusing fib({}). ~~~'.format(step))
			return self._fib_mesh[step]
		elif memoise:
			if step >= max_:
				self._fib_mesh[step] = fib_mesh(self.img, max_)
				return self._fib_mesh[step]
			elif step == max_ - 1:
				self._fib_mesh[step] = fib_mesh(self.img, step)
				return self._fib_mesh[step]
			one = self.fib_cache(step + 1, lock, True)
			two = self.fib_cache(step + 2, lock, True)
			self._fib_mesh[step] = np.append(one, two, axis=0)
			return self._fib_mesh[step]
		else:
			# no memoisation: one time transaction!
			self._fib_mesh[step] = fib_mesh(self.img, step)
			return self._fib_mesh[step]	
		
	
	def lloyd_cache(self, cell_itter, lock):
		'''
		Compute the lloyd mesh of (cell,itter) tuple, for current image and cashes it.
		'''
		lock.acquire()
		if self._lloyd_mesh.get(cell_itter) is None:
			self._lloyd_mesh[cell_itter] = lloyd_mesh(self.img, *cell_itter)
		lock.release()
		return self._lloyd_mesh[cell_itter]
