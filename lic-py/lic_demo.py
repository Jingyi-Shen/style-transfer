import numpy as np
import pylab as plt

import lic_internal

dpi = 100
size = 256

vortex_spacing = 0.2
extra_factor = 1.

a = np.array([1,0])*vortex_spacing
b = np.array([np.cos(np.pi/3),np.sin(np.pi/3)])*vortex_spacing
rnv = int(2*extra_factor/vortex_spacing)
vortices = [n*a+b*m for n in range(-rnv,rnv) for m in range(-rnv,rnv)]
vortices = [(x,y) for (x,y) in vortices if -extra_factor<x<extra_factor and -extra_factor<y<extra_factor]

xs = np.linspace(-1,1,size).astype(np.float32)[None,:]
ys = np.linspace(-1,1,size).astype(np.float32)[:,None]

vectors = np.zeros((size,size,2),dtype=np.float32)
for (x,y) in vortices:
    rsq = (xs-x)**2+(ys-y)**2
    print (xs-x), " ,",((xs-x)**2), rsq, x, xs
    # vectors[...,0] += y/rsq
    # vectors[...,1] += x/rsq
    vectors[...,0] += (ys-y)/rsq#(np.sin(x)+np.sin(y))
    vectors[...,1] += (xs-x)/rsq#(np.sin(x)-np.sin(y))
    print "vectors",vectors
    
texture = np.random.rand(size,size).astype(np.float32)
# texture = np.full((size,size), 0.1).astype(np.float32)
# np.zeros((size,size)).astype(np.float32)#
# plt.bone()
plt.gray()

kernellen=31
kernel = np.sin(np.arange(kernellen)*np.pi/kernellen)
kernel = kernel.astype(np.float32)

image = lic_internal.line_integral_convolution(vectors, texture, kernel)
print image,"image"
plt.clf()
plt.axis('off')
plt.figimage(image)
plt.gcf().set_size_inches((size/float(dpi),size/float(dpi)))
plt.savefig("images/flow-image5.png",dpi=dpi)



