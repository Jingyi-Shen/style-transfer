from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform


import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as Data  
from torchvision import transforms, utils
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision.utils import make_grid

import time
from datetime import datetime
import pickle

# class Hyper_Parameter:
#     def __init__(self):
#         self.cuda = False #use cuda?
#         self.train = True
#         self.train_batchSize = 10 #training batch size
#         self.val_batchSize = 10 #validation batch size
#         self.train_epochs = 100 #500 #number of epochs to train for
#         self.lr = 1e-3 #Learning Rate. Default=1e-3
#         self.weight_decay = 5**(-4) #weight decay
#         self.seed = 123 #random seed to use. Default=123
#         self.root = 'images/'
#         self.train_images = 'lic_4shape/'
#         # in OSC, i actually use gray images directly
#         # self.label_folderName = 'label'
#         self.val_images = 'gray-256/'
#         self.img_size = 256

class Hyper_Parameter:
    def __init__(self):
    	self.cuda = False
    	self.seed = 123 
        self.root = 'images/'
        self.train_images = 'lic_4shape-test/'
        self.val_images = 'gray-256/'
        self.img_size = 256
        self.lr = 1e-3
        self.weight_decay = 5**(-4)


def to_img(x):
    # x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, hp.img_size, hp.img_size)
    print ('x, x.shape',x.shape, x)
    return x

def read_image(path, index=0):
    """
        Args:
            path (string): Path to the image(train/testing) folder.
    """
    # images_dir = os.listdir(path)
    images_dir = [f for f in os.listdir(path) if not f.startswith('.')]
    # if images_dir[index].endswith('.jpg') or images_dir[index].endswith('.png'):
    image = io.imread(path+images_dir[index])#, as_gray=True)  # as_gray=False
    # print("image gray>>>>>>>>>>>>", image, image.shape)
    return image


class LICDataset(Dataset):
    def __init__(self, transform=None, isTrain=True):
        """
        Args:
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = transform
        self.isTrain = isTrain

    def __len__(self):
        if self.isTrain:
            # print("train", len(os.listdir(hp.root+hp.train_images))-1 )
            return len(os.listdir(hp.root+hp.train_images))-1 # For local MAC: ignore ./ds_store
        else:
            # print("val", len(os.listdir(hp.root+hp.val_images))-1 )
            return len(os.listdir(hp.root+hp.val_images))-1

    def __getitem__(self, idx):
        if self.isTrain:
            img_name = os.path.join(hp.root, hp.train_images)
        else:
            img_name = os.path.join(hp.root, hp.val_images)
        # image = io.imread(img_name) #[:, :, 0:3]
        image = read_image(img_name, idx)

        if self.transform:
            image = self.transform(image)
        # print("image>>>>>>>>>>>>", image, image.shape)
        return image

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, image):
        # numpy image: H x W x C
        # torch image: C X H X W
        # image = torch.from_numpy(image.transpose((2, 0, 1))).float()
        image = torch.from_numpy(image).float()
        return image

class Normalize(object):
    def __call__(self, image):
        # image = np.array(image, dtype=np.float64).flatten()
        # image = (image.astype(np.float32) - 127.5) / 127.5
        image = image.astype(np.float32)  / 255.0
        return image


class AutoEncoder(nn.Module):
    """AutoEncoder"""
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(16)
        # self.conv5 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1, bias=False)
        # self.bn5 = nn.BatchNorm2d(16)

        #self.fc1 = nn.Linear(16*16*16, 256)
        #self.fc2 = nn.Linear(16*8*8, 16*8*8)
        #self.upsample1 = nn.Linear(256, 16*16*16)
        #self.upsample2 = nn.Linear(1024, 16*16*16)

        # self.conv6 = nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        # self.bn6 = nn.BatchNorm2d(16)
        self.conv7 = nn.ConvTranspose2d(16, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(64)
        self.conv8 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(32)
        self.conv9 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(16)
        self.conv10 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()


    def forward(self, x, lat):
        conv1 = self.tanh(self.bn1(self.conv1(x)))
        conv2 = self.tanh(self.bn2(self.conv2(conv1)))
        conv3 = self.tanh(self.bn3(self.conv3(conv2)))
        conv4 = self.tanh(self.bn4(self.conv4(conv3))) # 16, 16, 16
        
        # temp = conv4.view(x.size(0), 16*16*16)
        # lat = (self.fc1(temp)) 
        # upsample = self.tanh(self.upsample1(lat))

        # Encoder(new latent vectors)
        lat = lat.view(-1, 16, 16, 16)
        conv7 = self.tanh(self.bn7(self.conv7(lat)))
        conv8 = self.tanh(self.bn8(self.conv8(conv7)))
        conv9 = self.tanh(self.bn9(self.conv9(conv8)))
        out = (self.conv10(conv9)).view(-1, 1, 256, 256)
        return out, conv4
        

class LatentEncoder(nn.Module):
    """AutoEncoder"""
    def __init__(self):
        super(LatentEncoder, self).__init__()
        #self.fc1 = nn.Linear(16*16*16, 256*8)
        #self.fc2 = nn.Linear(256*8, 256)
        self.upsample1 = nn.Linear(256, 256*8)
        self.upsample2 = nn.Linear(256*8, 16*16*16)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, lat_in):
        # temp = conv4.view(x.size(0), 16*16*16)
        # fc1 = self.tanh(self.fc1(lat_in))
        # lat = self.tanh(self.fc2(fc1))
        
        upsample1 = self.tanh(self.upsample1(lat_in))
        upsample2 = self.tanh(self.upsample2(upsample1))
        reshape = upsample2.view(-1, 16, 16, 16)
        return reshape


start = datetime.now()
hp = Hyper_Parameter()

model = AutoEncoder()
model.load_state_dict(torch.load('./AutoEn.pth',map_location='cpu'))
print(model)  # net architecture

# model.apply(init_weights)

model_lat = LatentEncoder()
model_lat.load_state_dict(torch.load('./Lat_AutoEn.pth',map_location='cpu'))

cuda = hp.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(hp.seed)
if cuda:
    torch.cuda.manual_seed(hp.seed)

loss_funcBCE = nn.BCEWithLogitsLoss() #
loss_funcMSE = nn.MSELoss()  # avg loss, scalar
optimizer = torch.optim.Adam(
    model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)

if cuda:
    model = model.cuda()
    model_lat = model_lat.cuda()
    loss_funcBCE = loss_funcBCE.cuda()
    loss_funcMSE = loss_funcMSE.cuda()


path = os.path.join(hp.root, hp.train_images)
images_dir = [f for f in os.listdir(path) if not f.startswith('.')]


with open('lat_new.pkl', 'rb') as file:
    dict_lat =pickle.load(file)

for num in range(len(images_dir)):
    image = io.imread(path+images_dir[num])
    image = torch.from_numpy(image).float() / 255.0
    image = image.view(-1, 1, hp.img_size, hp.img_size) 
    image = Variable(image) 
    if cuda:
        image = image.cuda()

    print('*' * 50, num)
    print('===> begin')
    since = datetime.now()

    lat_in = Variable(dict_lat[str(num)]) 
    if cuda:
        lat_in = lat_in.cuda()
    #reshape =  (lat_in) # forgot why I add this here
    output, conv4 = model(image, lat_in) 
    # print ("lat----------: ", lat_in, lat_in.shape)
    
    output = F.sigmoid(output)
    loss1 = loss_funcMSE(output , image) 
    loss2 = loss_funcMSE(conv4, reshape)
    loss = loss1+loss2
    print('loss_funcMSE(output , image) , loss_funcMSE(conv4, reshape)')
    print(loss1, loss2)

    with open(os.path.join('log_train_AutoEn.csv'),'a') as f:
        elapsed_time = (datetime.now() - since).total_seconds()
        log = [num] +['']*2 + [elapsed_time] + [loss] + [loss1] + [loss2]
        log = map(str, log)
        f.write(','.join(log) +'\n')
        print("=== pic {} Complete: Loss: {:.4f}".format(num, loss))

    if cuda:
        pic = to_img(output.cpu().data)
    else:
        pic = to_img(output.data)
    save_image(make_grid(pic, nrow=5), './result_fake/{}.png'.format(num))

# train()
end = datetime.now()
h, remainder = divmod((end - start).seconds, 3600)
m, s = divmod(remainder, 60)
time_str = 'Total Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
print (time_str)


		