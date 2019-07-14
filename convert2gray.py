# from PIL import Image
# import os
# from skimage import io, transform

# path = 'images/lic_imgs-256/'
# savapath = 'images/gray-256/'
# images_dir = [f for f in os.listdir(path) if not f.startswith('.')]
#     #images_dir.sort()
#     # if images_dir[index].endswith('.jpg') or images_dir[index].endswith('.png'):
#     for index in range(1000):
#         img = Image.open(path+images_dir[index]).convert('L')
#         img.save(savapath)
from PIL import Image
import os
 
input_dir = 'images/lic_4shape/'
out_dir = 'images/lic_4shape-gray/'
# a = os.listdir(input_dir)
images_dir = [f for f in os.listdir(input_dir) if not f.startswith('.')]

for i in images_dir:
    print(i)
    I = Image.open(input_dir+i)
    L = I.convert('L')
    L.save(out_dir+i)
