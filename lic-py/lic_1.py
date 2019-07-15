import numpy as np
import pylab as plt

import lic_internal
import random


dpi = 100
size = 256
EPSILON = 1e-10
vortex_spacing = 0.3
extra_factor = 2.

xs = np.linspace(0,1,size).astype(np.float32)[None,:]
ys = np.linspace(0,1,size).astype(np.float32)[:,None]

def sink(r, center_x, center_y):
	vortices = [(x,y) for x in range(0, size, 5) for y in range(0, size, 5)]
	vortices = [(x,y) for (x,y) in vortices if (center_x-x)**2<r**2 and (center_y-y)**2<r**2]
	for (x,y) in vortices:
		x=x/256.0
		y=y/256.0
		rsq = (x-xs)**2+(y-ys)**2
		# vectors[x,y,0] += y/rsq
		# vectors[x,y,1] += -x/rsq
		# rsq = np.sqrt((center_x-x)**2+(center_y-y)**2)+ EPSILON
		vectors[...,0] += -(center_y/256.0-y)/rsq
		vectors[...,1] += -(center_x/256.0-x)/rsq


def source(r, center_x, center_y):
	vortices = [(x,y) for x in range(0, size, 5) for y in range(0, size, 5)]
	vortices = [(x,y) for (x,y) in vortices if (center_x-x)**2<r**2 and (center_y-y)**2<r**2]

	for (x,y) in vortices:
		x=x/256.0
		y=y/256.0
		rsq = (x-xs)**2+(y-ys)**2
		# vectors[x,y,0] += y/rsq
		# vectors[x,y,1] += -x/rsq
		# rsq = np.sqrt((center_x-x)**2+(center_y-y)**2)+ EPSILON
		vectors[...,0] += (center_y/256.0-y)/rsq
		vectors[...,1] += (center_x/256.0-x)/rsq

	

def counter_clock_wise(r, center_x, center_y):
	vortices = [(x,y) for x in range(0, size, 5) for y in range(0, size, 5)]
	vortices = [(x,y) for (x,y) in vortices if (center_x-x)**2<r**2 and (center_y-y)**2<r**2]
	
	for (x,y) in vortices:
		x=x/256.0
		y=y/256.0
		rsq = (x-xs)**2+(y-ys)**2
		# vectors[x,y,0] += y/rsq
		# vectors[x,y,1] += -x/rsq
		# rsq = np.sqrt((center_x-x)**2+(center_y-y)**2)+ EPSILON
		vectors[...,0] += (center_y/256.0-y)/rsq
		vectors[...,1] += -(center_x/256.0-x)/rsq
	
def clock_wise(r, center_x, center_y):
	vortices = [(x,y) for x in range(0, size, 5) for y in range(0, size, 5)]
	# vortices = [(x,y) for (x,y) in vortices if 0<x<size and 0<y<size]
	vortices = [(x,y) for (x,y) in vortices if (center_x-x)**2<r**2 and (center_y-y)**2<r**2]
	
	for (x,y) in vortices:
		x=x/256.0
		y=y/256.0
		rsq = (x-xs)**2+(y-ys)**2
		# vectors[x,y,0] += y/rsq
		# vectors[x,y,1] += -x/rsq
		# rsq = np.sqrt((center_x-x)**2+(center_y-y)**2)+ EPSILON
		vectors[...,0] += -(center_y/256.0-y)/rsq
		vectors[...,1] += (center_x/256.0-x)/rsq

# random_center = 2
# for i in range(random_center):
# r = 30
num = 1000
for ii in range(num):
	r = random.randint(30, 60)
	print ii, "r", r
	vectors = np.zeros((size,size,2),dtype=np.float32)+EPSILON

	t=random.random()
	if(t<0.5):
		print("clock_wise")
		centerx = np.floor(random.randint(1, size/2)).astype(np.int)#extra_factor/vortex_spacing
		centery = np.floor(random.randint(1, size/2)).astype(np.int)#extra_factor/vortex_spacing
		# print("clock_wise centery, centerx 1:", centerx, centery)
		clock_wise(r, centerx, centery)

	t=random.random()
	if(t<0.5):
		print("source")
		centerx = np.floor(random.randint(size/2+1, size-1)).astype(np.int)
		centery = np.floor(random.randint(1, size/2)).astype(np.int)
		# print("source centery, centerx 3:", centerx, centery)
		source(r, centerx, centery)

	t=random.random()
	if(t<0.5):
		print("counter_clock_wise")
		centerx = np.floor(random.randint(1, size/2)).astype(np.int)
		centery = np.floor(random.randint(size/2+1, size-1)).astype(np.int)
		# print("counter_clock_wise centery, centerx 2:", centerx, centery)
		counter_clock_wise(r, centerx, centery)

	t=random.random()
	if(t<0.5):
		print("sink")
		centerx = np.floor(random.randint(size/2+1, size-1)).astype(np.int)
		centery = np.floor(random.randint(size/2+1, size-1)).astype(np.int)
		# print("sink centery, centerx 4:", centerx, centery)
		sink(r, centerx, centery)

	texture = np.random.rand(size,size).astype(np.float32)
	plt.gray()

	kernellen=50
	kernel = np.sin(np.arange(kernellen)*np.pi/kernellen)
	kernel = kernel.astype(np.float32)

	image = lic_internal.line_integral_convolution(vectors, texture, kernel)

	plt.clf()
	plt.axis('off')
	plt.figimage(image)
	plt.gcf().set_size_inches((size/float(dpi),size/float(dpi)))
	plt.savefig("images_rand_shape/"+str(ii)+".png",dpi=dpi)


