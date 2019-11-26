# Tatiana Zihindula
# C16339923
# "Group" Project
# 23/11/2019

# All the code in here is just to display the output in a seamless fast way.
# some images are high resolutions other are not.
# low res images are fast, but high res images take longer.
# all the concurent stuff here just allow to run  all images in parallel so the wait is short
# please ignore the thread/multiprocessing/lock/queue part of it.
# or maybe just glance/fast scroll
# the CPU might scream abit because of the 'all of the sudden' work it has to do, but thatere is a ton of it

import cv2, random, glob, time
from queue import Queue
from threading import Thread
import multiprocessing
from functools import reduce

# local imports

from tiles import squares, pentagons, triangles, circles
from mesh import MeshCache
from stylise import map_gradient
from util import show_fewer


# IMPORTANT: if running on OSX/Mojave/Catalina/etc.
# and for some Apple's security reasons Fork form outside the main thread is being picky
# run the env bellow.

# export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

# https://github.com/ansible/ansible/issues/32499


# NOTE: one procss has to finish at least all processes before it displays them
# now that I am thinking I should have used semaphores.

# more mosaic tiles: https://tinyurl.com/wsxc59w


import glob 
if __name__=='__main__':
	
	# read all the images in the image folders
	imgs = glob.glob('./images/*[.jpg, .png]')

	# shuffle to images, to be surprised everytime
	random.shuffle(imgs)

	# TEST CASES
	funcs = {
		'squares': [
			(squares,{}),
			(squares,{'s':5}),
			(squares,{'s':30}),
			(squares,{'s':50}),
		],
		'triangles': [
			(triangles,{}),
			(triangles,{'fib_step':7}),
			(triangles,{'fib_step':11}),
			(triangles,{'fib_step':17}),
			(triangles,{'fib_step':31}),
		],
		'circles': [
			(circles,{}),
			(circles,{'s':.5}),
			(circles,{'s':2}),
			(circles,{'s':7}),
		],
		'pentagons': [
			(pentagons,{}),
			(pentagons,{'fib_step':7}),
			(pentagons,{'fib_step':23}),
			(pentagons,{'lloyd_cells':2}),
			(pentagons,{'lloyd_cells':47}),
			(pentagons,{'lloyd_cells':97}),
		],
	}
	# the number of different type of mosaics to execute. this is the same size as the test cases.
	FUNCS_COUNT = reduce(lambda x,y: x+y, map(lambda x: len(funcs[x]), funcs))
	MAX_INFLIGHT = 3
	sem = multiprocessing.Semaphore(value=MAX_INFLIGHT)
	def make_mosaics_worker(im, q_display):
		sem.acquire()
		q_mosaics = Queue() # when all mosaics are done processing they will be here
		q_gradients = Queue() # same for gradients
		mesh_cache = MeshCache(im) # one mesh cache per image
			
		fib_mesh_lock = multiprocessing.Lock() # when getting the fib mesh cache
		lloyd_mesh_lock = multiprocessing.Lock() # when getting the lloyd mesh cache
		
		# for all operations needed to be done on the image, create a thread for them
		for tile_type in funcs:
			# for each function, get its argument as predefined
			for f, kwargs in funcs[tile_type]:
				# when each finishes, set the images in the q_mosaics
				step = kwargs.get('fib_step') # 5% of the image sampling looks good play with this variable
				cells = kwargs.get('lloyd_cells') # e.g. is 2 = 50%, 5 = 20% dense
				itter = 10 # find the perfect lloyd tiles after 10 itterations
				
				kwargs.update({'queue': q_mosaics})
				# if the current function requires a mesh, get the cached one
				# or compute it if fisrt time.
				if kwargs.get('fib_step'):
					kwargs.update({'points': mesh_cache.fib_cache(step,fib_mesh_lock)})
				elif kwargs.get('lloyd_cells'):
					kwargs.update({'points': mesh_cache.lloyd_cache((cells,itter),lloyd_mesh_lock)})
				# start a thread for this tile type
				Thread(target=f, args=(im,), kwargs=kwargs).start()
				
		print('--finished sending all funcs--')

		# temporary storage before they go to be displayed
		# optionally they would be sent to the display queue, but
		# it;s expecting them in bashes, so.. this is the bash.
		mosaics = []
		for i in range(FUNCS_COUNT):
			# so for every type of mosaic and its variations, collect them form the queue
			# this will block until there is something in the queue.
			img = q_mosaics.get()
			# start a thread that will convert this image to gradient.
			Thread(target=map_gradient, args=(img[1],),kwargs={'queue': q_gradients}).start()
			# don't wait for any of it. you are done here.
			mosaics.append(img)
			q_mosaics.task_done()
		# wait for all mosaic thread to put something in the queue.
		q_mosaics.join()
		print('---finished all mosaics---')
		
		gradients = []
		for i in range(FUNCS_COUNT):
			# gradients.get() will block until there is a gradient there to collect
			# NOTE: all this is intended.
			gr = q_gradients.get()
			# when a gradient has been appended we are done here.
			gradients.append(gr)
			q_gradients.task_done()
			
		# wait for all gradients to finish. kinda unecessary Queue.get() block if nothing is there
		# but for best practice's sake//
		q_gradients.join()
		print('\n~~~finished all gradients~~~\n')

		# send everything at once to the display queue
		q_display.put((im, mosaics, gradients))
		# BUG: the first one shouldn't sleep. the rest, yes. maybe have a done processing variable?
		# TODO: on btw you can use that processing variable to make a progressbar? that'd be cool..
		# the wait time for each subsequenet should also be proportional to FUNC_COUNT
		# and not just some majic number 20
		print('\nSLEEPING~~~\n')
		time.sleep(20)
		print('\nDONE~~~\n')
		sem.release()
		
		
	# -- THE START ---

	show_lock = multiprocessing.Lock() # to only one image to show its variation at a time
	q_display = multiprocessing.Queue() # while one image is displaying wait here
	IMGS_NUM = len(imgs) # number of images to work with only display a few
	
	for i in range(IMGS_NUM):
		# get the current image
		im = cv2.imread(imgs[i])
		# start a processes to process the current image concurently
		print('starting process for img:{} {}'.format(i,imgs[i]))
		multiprocessing.Process(target=make_mosaics_worker, args=(im, q_display)).start()
			
	# display the images
	for i in range(IMGS_NUM):
		
		show_lock.acquire()
		img, mosaics, gradients = q_display.get()
		print('SHOO MMEEE WHAAT-ACHOU YOU GUUTTT !!')
		
		show_fewer(img, mosaics, gradients, l=0, h=FUNCS_COUNT)

		# some references from Rick and Morty to get me excited when I look at the logs on stdout
		dice = random.randint(0,2)
		if dice == 1:
			print('I LUWIKE WHART YOU GUUUT!!')
		elif dice == 2:
			print('hmmm--- (DISQUALIFIEDDD!!)')
		else:
			print('I DON\'T LUWIKE WHAAT-ACHOU GUUUT!!')
		show_lock.release()
			
	q_display.close()
	q_display.join_thread()
