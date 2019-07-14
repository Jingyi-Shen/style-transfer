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

class Hyper_Parameter:
    def __init__(self):
        self.cuda = True #use cuda?
        self.train = True
        self.train_batchSize = 20 #training batch size
        self.val_batchSize = 10 #validation batch size
        self.train_epochs = 100 #500 #number of epochs to train for
        self.lr = 1e-3 #Learning Rate. Default=1e-3
        self.weight_decay = 5**(-4) #weight decay
        self.seed = 123 #random seed to use. Default=123
        self.root = '/users/PAS0027/shen1250/Project/lic/images/'
        self.train_images = 'lic_4shape-gray/'
        # in OSC, i actually use gray images directly
        # self.label_folderName = 'label'
        self.val_images = 'gray-256/'
        self.img_size = 256

# def imshow(image):
#   plt.imshow(image.numpy().transpose((1, 2, 0)))

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
            return len(os.listdir(hp.root+hp.train_images)) # ignore ./ds_store
        else:
            # print("val", len(os.listdir(hp.root+hp.val_images))-1 )
            return len(os.listdir(hp.root+hp.val_images))

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

# class Resize(object):
#     def __init__(self, size):
#         assert isinstance(size, (int, tuple))
#         self.size = size

#     def __call__(self, image):
#         h, w = image.shape[:2]
        
#         if isinstance(self.size, int):
#             if h > w:
#                 new_h, new_w = self.size * h / w, self.size
#             else:
#                 new_h, new_w = self.size, self.size * w / h
#         else:
#             new_h, new_w = self.size

#         new_h, new_w = int(new_h), int(new_w)

#         image = transform.resize(
#             image, (new_h, new_w), order=1, mode="reflect",
#             preserve_range=True, anti_aliasing=True).astype(np.float32)
#         # print('after resize, ',image.shape)


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

        self.fc1 = nn.Linear(16*16*16, 256)
        self.fc2 = nn.Linear(16*8*8, 16*8*8)

        # Decoder
        self.upsample1 = nn.Linear(256, 16*16*16)
        self.upsample2 = nn.Linear(1024, 16*16*16)

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


    def forward(self, x):
        conv1 = self.tanh(self.bn1(self.conv1(x)))
        conv2 = self.tanh(self.bn2(self.conv2(conv1)))
        conv3 = self.tanh(self.bn3(self.conv3(conv2)))
        conv4 = self.tanh(self.bn4(self.conv4(conv3))) # 16, 16, 16
        
        # temp = conv4.view(x.size(0), 16*16*16)
        # lat = (self.fc1(temp)) 
        # upsample = self.tanh(self.upsample1(lat))
        # reshape = upsample.view(-1, 16, 16, 16) 
        
        conv7 = self.tanh(self.bn7(self.conv7(conv4)))
        conv8 = self.tanh(self.bn8(self.conv8(conv7)))
        conv9 = self.tanh(self.bn9(self.conv9(conv8)))
        out = (self.conv10(conv9)).view(-1, 1, 256, 256)
        return out, conv4


class LatentEncoder(nn.Module):
    """AutoEncoder"""
    def __init__(self):
        super(LatentEncoder, self).__init__()
        self.fc1 = nn.Linear(16*16*16, 256*8)
        self.fc2 = nn.Linear(256*8, 256)
        self.upsample1 = nn.Linear(256, 256*8)
        self.upsample2 = nn.Linear(256*8, 16*16*16)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        temp = conv4.view(x.size(0), 16*16*16)
        fc1 = self.tanh(self.fc1(temp))
        lat = self.tanh(self.fc2(fc1))
        
        upsample1 = self.tanh(self.upsample1(lat))
        upsample2 = self.tanh(self.upsample2(upsample1))
        reshape = upsample2.view(-1, 16, 16, 16)
        return lat, reshape


start = datetime.now()
hp = Hyper_Parameter()

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

model = AutoEncoder()
print('model',model)  # net architecture
# model.load_state_dict(torch.load('/users/PAS0027/shen1250/Project/lic/AutoEn2_99.pth'))
model.apply(init_weights)

model_lat = LatentEncoder()
print('model_lat', model_lat)
model_lat.apply(init_weights)

cuda = hp.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(hp.seed)
if cuda:
    torch.cuda.manual_seed(hp.seed)

print('===> Loading LIC datasets')

train_dataset = LICDataset(isTrain=True,transform=transforms.Compose([Normalize(), ToTensor()]))
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=hp.train_batchSize, shuffle=True)

val_dataset = LICDataset(isTrain=False,transform=transforms.Compose([Normalize(), ToTensor()]))
val_loader = Data.DataLoader(dataset=val_dataset, batch_size=hp.val_batchSize, shuffle=False)


loss_funcBCE = nn.BCEWithLogitsLoss() #
loss_funcMSE = nn.MSELoss()  # avg loss, scalar
optimizer = torch.optim.Adam(
    model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)

if cuda:
    model = model.cuda()
    model_lat = model_lat.cuda()
    loss_funcBCE = loss_funcBCE.cuda()
    loss_funcMSE = loss_funcMSE.cuda()

trainLoss_=[]
trainLoss_1=[]
trainLoss_2=[]
valLoss_=[]

for epoch in range(hp.train_epochs):
    print('*' * 50)
    print('epoch {}'.format(epoch+1))

    print('===> training begin')
    since_train = datetime.now()
    model = model.train()
    train_loss = 0
    train_loss1 =0
    train_loss2 =0
    val_loss = 0

    for iteration, sample_batched in enumerate(train_loader): 
        since_iter = datetime.now()
        sample_batched = sample_batched.view(-1, 1, hp.img_size, hp.img_size) 
        # print(sample_batched.shape," sample_batched shape")
        
        img = Variable(sample_batched)   # batch x
        if cuda:
            img = img.cuda()

        # ===================forward=====================
        # print('>>>', datetime.now(), 'forward')
        output, conv4 = model(img) 
        lat, reshape = model_lat(conv4)

        for i in range(16):
            # lat_ = lat[:, i, :, :]*100
            lat_ = reshape[:, i, :, :]
            print("lat_.max() ", lat_.max(), lat_.min())
            k = 1.0 / (lat_.max() - lat_.min());
            lat_ = k * (lat_ - lat_.min())

            pic_lat = to_img(lat_, 16, 16)

            save_image(make_grid(pic_lat, nrow=5), './result_reshape/{}_{}.png'.format(iteration, i))

        for i in range(16):
            # lat_ = lat[:, i, :, :]*100
            lat_ = conv4[:, i, :, :]
            print("lat_.max() ", lat_.max(), lat_.min())
            k = 1.0 / (lat_.max() - lat_.min());
            lat_ = k * (lat_ - lat_.min())

            pic_lat = to_img(lat_, 16, 16)

            save_image(make_grid(pic_lat, nrow=5), './result_conv4/{}_{}.png'.format(iteration, i))

        # output = F.sigmoid(output)    
        print ("lat----------: ", lat, lat.shape)
        print ("output----------: ", output)
        print ("img----------: ", img)
        output = F.sigmoid(output)
        # print ("sigmoid output----------: ", output)
        
        loss1 = loss_funcMSE(output , img) 
        loss2 = loss_funcMSE(conv4, reshape)
        loss = loss1+loss2
        print('loss_funcMSE(output , img) , loss_funcMSE(conv4, reshape)')
        print(loss1, loss2)

        trainLoss_.append(loss.item())
        trainLoss_1.append(loss1.item())
        trainLoss_2.append(loss2.item())
        # print ('>>>', datetime.now(), 'output: ', output, output.shape, 'get predicition, loss is:', loss.item())
        train_loss += loss.item()
        train_loss1 += loss1.item()
        train_loss2 += loss2.item()
        # ===================backward====================
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients

        optimizer.step()                # apply gradients

        print('epoch [{}-{}/{}], loss:{:.4f}, time:{}'.format(epoch+1, iteration, hp.train_epochs, loss.item(), datetime.now()-since_iter))
        # plt.plot(range(epoch*hp.train_batchSize+iteration), trainLoss_[:epoch*hp.train_batchSize+iteration],'g-') #,label='total loss')
        # plt.plot(range(epoch*hp.train_batchSize+iteration), trainLoss_1[:epoch*hp.train_batchSize+iteration], 'b-') #,label='MSE(output, img)')
        # plt.plot(range(epoch*hp.train_batchSize+iteration), trainLoss_2[:epoch*hp.train_batchSize+iteration], 'r-') #,label='MSE(conv4, reshape)')

    with open(os.path.join('/users/PAS0027/shen1250/Project/lic/log_train_AutoEn3_100.csv'),'a') as f:
        avg_loss = train_loss / len(train_loader)
        avg_loss1 = train_loss1 / len(train_loader)
        avg_loss2 = train_loss2 / len(train_loader)
        # elapsed_time = (datetime.now() - since_train).total_seconds()
        log = [epoch, iteration] +['']*2 + [avg_loss1] + [avg_loss2] + [avg_loss]
        log = map(str, log)
        f.write(','.join(log) +'\n')
        print("=== Epoch {} Complete: Avg.Train Loss: {:.4f}".format(epoch, avg_loss))
    
    if cuda:
        pic = to_img(output.cpu().data)
    else:
        pic = to_img(output.data)

    torch.save(model.state_dict(), '/users/PAS0027/shen1250/Project/lic/AutoEn3_'+str(epoch)+'.pth')
    torch.save(model_lat.state_dict(), '/users/PAS0027/shen1250/Project/lic/Lat_AutoEn3_'+str(epoch)+'.pth')
    # torch.save(model, './full_AutoEn_1_'+str(epoch)+'.pth')
    save_image(make_grid(pic, nrow=5), '/users/PAS0027/shen1250/Project/lic/result_7/{}.png'.format(epoch))

# train()
end = datetime.now()
h, remainder = divmod((end - start).seconds, 3600)
m, s = divmod(remainder, 60)
time_str = 'Total Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
print (time_str)


