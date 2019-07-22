import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os, glob
import scipy.io as sio

# data_dir = '/home/cvnlp/deepfashion/Category_and_Attribute_Prediction_Benchmark_2/'  # train
features_dir = '/home/cvnlp/tiantian/project01/data/ddsm/dl_feature_mat'  # Resnet_features_train

# // 这里自己修改网络


class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.net = models.resnet152(pretrained=True)

    def forward(self, input):
        output = self.net.conv1(input)
        output = self.net.bn1(output)
        output = self.net.relu(output)
        output = self.net.maxpool(output)
        output = self.net.layer1(output)
        output = self.net.layer2(output)
        output = self.net.layer3(output)
        output = self.net.layer4(output)
        output = self.net.avgpool(output)
        return output


model = net()
# // 加载cuda
model = model.cuda()

extensions = ['jpg', 'jpeg', 'JPG', 'JPEG', 'png']
features = []
files_list = []
imgs_path = open("/home/cvnlp/tiantian/project01/data/ddsm/pathname_ddsm.txt").read().splitlines()
# x = os.walk(data_dir)
# for path, d, filelist in x:
#     for filename in filelist:
#         file_glob = os.path.join(path, filename)
#         files_list.extend(glob.glob(file_glob))
#
# print(files_list)
for i, img in enumerate(imgs_path):
    print("%d %s" % (i, img))
print("")
use_gpu = torch.cuda.is_available()
# for x_path in files_list:
#     print("x_path" + x_path)
#     file_name = x_path.split('/')[-1]
#     fx_path = os.path.join(features_dir, file_name + '.txt')
    # print(fx_path)
    # extractor(x_path, fx_path, model, use_gpu)

for i, im in enumerate(imgs_path):
# def extractor(img_path, saved_path, net, use_gpu):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()]
    )

    img = Image.open(im)
    img = transform(img)
    print(im)
    print(img.shape)

    x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
    print(x.shape)

    if use_gpu:
        x = x.cuda()
        model = model.cuda()
    y = model(x).cpu()
    y = torch.squeeze(y)
    y = y.data.numpy()
    print(y.shape)
    # np.savetxt(saved_path, y, delimiter=',')
    feature = np.reshape(y, [1, -1])
    features.append(feature)
features = np.array(features)
dic = {'ddsm_resnet152': features}
sio.savemat(features_dir + '/ddsm_resnet152' + '.mat', dic)