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
        self.cuda = False #use cuda?
        self.train = True
        self.train_batchSize = 20 #training batch size
        self.val_batchSize = 10 #testing batch size
        self.train_epochs = 300 #number of epochs to train for
        self.lr = 1e-3 #Learning Rate. Default=1e-3
        self.weight_decay = 5**(-4) #weight decay
        self.seed = 123 #random seed to use. Default=123
        self.root = 'images/'
        self.train_images = 'lic_imgs-256/'
        # in OSC, i actually use gray images directly
        # self.label_folderName = 'label'
        self.val_images = 'lic_val-256/'
        self.img_size = 256

# def imshow(image):
#   plt.imshow(image.numpy().transpose((1, 2, 0)))

def to_img(x):
    # x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, hp.img_size, hp.img_size)
    print ('x, x.shape',x.shape)
    return x

def read_image(path, index=0):
    """
        Args:
            path (string): Path to the image(train/testing) folder.
    """
    # images_dir = os.listdir(path)
    images_dir = [f for f in os.listdir(path) if not f.startswith('.')]
    #images_dir.sort()
    # if images_dir[index].endswith('.jpg') or images_dir[index].endswith('.png'):
    image = io.imread(path+images_dir[index], as_gray=True)  # as_gray=False
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
            return len(os.listdir(hp.root+hp.train_images))-1 # ignore ./ds_store
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
        return image

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, image):
        # numpy image: H x W x C
        # torch image: C X H X W
        # image = torch.from_numpy(image.transpose((2, 0, 1))).float()
        image = torch.from_numpy(image).float()
        # print ("ToTensor", image, image.shape)
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
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
        #     nn.ReLU(True),
        #     nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
        #     nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
        #     nn.ReLU(True),
        #     nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        # )
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
        #     nn.Tanh()
        # )

        # input_img = Input(shape=(img_width, img_height, 3))

        # # Encoding network
        # x = Conv2D(16, (3, 3), activation='relu', padding='same', strides=2)(input_img)
        # x = Conv2D(32, (3, 3), activation='relu', padding='same', strides=2)(x)
        # encoded = Conv2D(32, (2, 2), activation='relu', padding="same", strides=2)(x)
        # # Decoding network
        # x = Conv2D(32, (2, 2), activation='relu', padding="same")(encoded)
        # x = UpSampling2D((2, 2))(x)
        # x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        # x = UpSampling2D((2, 2))(x)
        # x = Conv2D(16, (3, 3), activation='relu')(x)
        # x = UpSampling2D((2, 2))(x)
        # decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

        # Encoder
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False) #(20, 16, 128, 128)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(16)

        # Decoder
        self.conv5 = nn.ConvTranspose2d(16, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(32)
        self.conv7 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(16)
        self.conv8 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        
        self.relu = nn.ReLU()


    def forward(self, x):
        # x = self.encoder(x)
        # x = self.decoder(x)
        # return x
        conv1 = self.relu(self.bn1(self.conv1(x)))
        conv2 = self.relu(self.bn2(self.conv2(conv1)))
        conv3 = self.relu(self.bn3(self.conv3(conv2)))
        conv4 = self.relu(self.bn4(self.conv4(conv3)))
        conv5 = self.relu(self.bn5(self.conv5(conv4)))
        conv6 = self.relu(self.bn6(self.conv6(conv5)))
        conv7 = self.relu(self.bn7(self.conv7(conv6)))
        out = self.conv8(conv7).view(-1, 1, 256, 256)
        return out


start = datetime.now()
hp = Hyper_Parameter()

model = AutoEncoder()
print(model)  # net architecture
# model.load_state_dict(torch.load('./AutoEn_2_590.pth', map_location='cpu'))

cuda = hp.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(hp.seed)
if cuda:
    torch.cuda.manual_seed(hp.seed)

print('===> Loading LIC datasets')

train_dataset = LICDataset(isTrain=True,transform=transforms.Compose([ToTensor()]))
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=hp.train_batchSize, shuffle=True)

val_dataset = LICDataset(isTrain=False,transform=transforms.Compose([ToTensor()]))
val_loader = Data.DataLoader(dataset=val_dataset, batch_size=hp.val_batchSize, shuffle=False)


loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)

if cuda:
    model = model.cuda()
    loss_func = loss_func.cuda()


trainLoss_=[]
valLoss_=[]

for epoch in range(hp.train_epochs):
    print('*' * 50)
    print('epoch {}'.format(epoch+1))

    print('===> training begin')
    since_train = datetime.now()
    model = model.train()
    train_loss = 0
    val_loss = 0

    for iteration, sample_batched in enumerate(train_loader): 
        since_iter = datetime.now()
        sample_batched = sample_batched.view(-1, 1, hp.img_size, hp.img_size) 
        print("sample_train batch", sample_batched.shape)

        img = Variable(sample_batched)   # batch x
        if cuda:
            img = img.cuda()

        # ===================forward=====================
        # print('>>>', datetime.now(), 'forward')
        output = F.sigmoid(model(img))    
        loss = loss_func(output, img)           # model output
        trainLoss_.append(loss.item())
        # print ('>>>', datetime.now(), 'output: ', output, output.shape, 'get predicition, loss is:', loss.item())
        train_loss += loss.item()
        # ===================backward====================
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        print('epoch [{}-{}/{}], loss:{:.4f}, time:{}'.format(epoch+1, iteration, hp.train_epochs, loss.item(), datetime.now()-since_iter))
        plt.plot(range(epoch*hp.train_batchSize+iteration), trainLoss_[:epoch*hp.train_batchSize+iteration])

    with open(os.path.join('log_train_AutoEn_1.csv'),'a') as f:
        avg_loss = train_loss / len(train_loader)
        elapsed_time = (datetime.now() - since_train).total_seconds()
        log = [epoch,iteration] +['']*2 + [elapsed_time] + [avg_loss]
        log = map(str, log)
        f.write(','.join(log) +'\n')
        print("=== Epoch {} Complete: Avg.Train Loss: {:.4f}".format(epoch, avg_loss))


    if epoch%10 is 0:
        if cuda:
            pic = to_img(output.cpu().data)
        else:
            pic = to_img(output.data)
        torch.save(model.state_dict(), './AutoEn_1_'+str(epoch)+'.pth')
        # torch.save(model, './full_AutoEn_1_'+str(epoch)+'.pth')
        save_image(make_grid(pic, nrow=5), './result/{}.png'.format(epoch))


    # print('===> validation begin')
    # for iteration, val_batches in enumerate(val_loader): 
    #     val_batch = val_batches.view(-1, 1, hp.img_size, hp.img_size) 
    #     print("val_batch", val_batch.shape)

    #     img = Variable(val_batch) 
    #     if cuda:
    #         img = img.cuda()

    #     output = F.sigmoid(model(img))    
    #     loss = loss_func(output, img)           # model output
    #     valLoss_.append(loss.item())
    #     val_loss += loss.item()
        
    #     print ('>>>', datetime.now(), 'output: ', output, output.shape, 'get predicition loss is:', loss.item())
    #     plt.plot(range(epoch*hp.val_batchSize+iteration),valLoss_[:epoch*hp.val_batchSize+iteration])
    
        
    avg_val_loss = val_loss / len(val_loader)
    print("=== Epoch {} Complete: Avg.Val Loss: {:.4f}".format(epoch, avg_val_loss))

    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Loss')

    plt.savefig('./loss_{}.png'.format(epoch))

# train()
end = datetime.now()
h, remainder = divmod((end - start).seconds, 3600)
m, s = divmod(remainder, 60)
time_str = 'Total Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
print (time_str)


