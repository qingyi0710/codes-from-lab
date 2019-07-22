import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os, glob
import scipy.io as sio
# import utils
# data_dir = 'F:/newkth2/KTH2/'  # train
features_dir = '/home/cvnlp/tiantian/project01/data/ddsm/dl_feature_mat'  # Vgg_features_train
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        VGG = models.vgg16(pretrained=True)
        self.feature = VGG.features
        self.classifier = nn.Sequential(*list(VGG.classifier.children())[:-3])
        pretrained_dict = VGG.state_dict()
        model_dict = self.classifier.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.classifier.load_state_dict(model_dict)

    def forward(self, x):
        output = self.feature(x)
        output = output.view(output.size(0), -1)
        output = self.classifier(output)
        return output
model = Encoder()
model = model.cuda()

extensions = ['jpg', 'jpeg', 'JPG', 'JPEG' ,'png']
features = []
files_list = []
imgs_path = open("/home/cvnlp/tiantian/project01/data/ddsm/pathname_ddsm.txt").read().splitlines()

# imgs_path = ["./2-PCS-Bags-in-1-Set-Fashion-PU-Leather-Women-Purses-and-Handbag-with-Coin-Bags-Set-YXL-240-.jpg",
#              "./4-Detachable-Pockets-PU-Golf-Bag.jpg",
#              "./6-Dividers-PU-Golf-Stand-Bag.jpg",
#              # "./img-tatoo-plane-224x224.jpg",
#              # "./img-dog-224x224.jpg",
#              # "./img-paper-plane-224x224.jpg",
#              # "./img-pyramid-224x224.jpg",
#              # "./img-tiger-224x224.jpg"
#              ]
# imgs = utils.load_images(*imgs_path)
# x = os.walk(data_dir)
# for path, d, filelist in x:
#     for filename in filelist:
#         file_glob = os.path.join(path, filename)
#         files_list.extend(glob.glob(file_glob))


# print(files_list)
use_gpu = torch.cuda.is_available()
# for x_path in files_list:
#     # print("x_path" + x_path)
#     file_name = x_path.split('/')[-1]
#     fx_path = os.path.join(features_dir, file_name + '.txt')
    # print(fx_path)
    # extractor(x_path, fx_path, model, use_gpu)
for i, im in enumerate(imgs_path):
    print("%d %s" % (i, im))
    print("")
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
    feature = np.reshape(y, [1, -1])
    features.append(feature)
features = np.array(features)
dic = {'ddsm_vgg16': features}
sio.savemat(features_dir + '/ddsm_vgg16' + '.mat', dic)