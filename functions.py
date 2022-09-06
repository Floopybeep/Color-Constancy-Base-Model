from PIL import Image

import os
import torch
import torchvision
import torch.nn as nn
from torchvision.transforms import *
import matplotlib.pyplot as plt

from options import opt


def load_img(filepath):
    img = Image.open(filepath)

    transform = transforms.Compose([
        transforms.PILToTensor()
    ])

    img_tensor = transform(img)
    return img_tensor


def angular_loss(self, pred, gt):
    a = (pred[:, 0] * gt[:, 0] + pred[:, 1] * gt[:, 1] + pred[:, 2] * gt[:, 2])
    b = pow(torch.abs(pred[:, 0] * pred[:, 0] + pred[:, 1] * pred[:, 1] + pred[:, 2] * pred[:, 2] + 1e-06), 1 / 2)
    c = pow(torch.abs(gt[:, 0] * gt[:, 0] + gt[:, 1] * gt[:, 1] + gt[:, 2] * gt[:, 2] + 1e-06), 1 / 2)

    acos = torch.acos(a / (b * c + 1e-06))
    output = torch.mean(acos)
    output = output * 180 / 3.14  # degree

    return output


class Aloss(nn.Module):
    def __init__(self):
        super(Aloss, self).__init__()

    def forward(self, out, gt):
        return angular_loss(self, out, gt)


def log(text, logger_file):
    with open(logger_file, 'a') as f:
        f.write(text)
        f.close()


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def save_img(img, directory, img_name):
    save_dir = opt.output
    save_fn = save_dir + '/' + directory + '/' + img_name
    if not os.path.exists(save_dir + '/' + directory):
        os.makedirs(save_dir + '/' + directory)

    torchvision.utils.save_image(img, save_fn, normalize=True, format='png')


def print_graph(train_loss, path):
    plt.plot(train_loss)
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss (degree)')
    plt.savefig(path + 'training_graph.png')


def image_normalize(image):
    max_x = image.shape[0]
    max_y = image.shape[1]
    max_int = 0

    for x in range(max_x):
        for y in range(max_y):
            if image[x, y] > max_int:
                max_int = image[x, y]

    for x in range(max_x):
        for y in range(max_y):
            image[x, y] = int(image[x, y] * 256 / (max_int + 0.1))

    return image


def print_feature_map(in_tensor, index, layer_name, normalize=True):
    (b, c, h, w) = in_tensor.shape                          # batch (16 per batch), channel, height, width
    in_tensor_np = in_tensor.cpu().detach().numpy()         # change input tensor into more accessible numpy array
    index_np = index.cpu().detach().numpy()                 # change index tensor into numpy array for data extraction
    index = index_np.item() + 1

    # print(b, c, h, w)

    for i in range(b):
        for j in range(c):
            # path : string for storage location of tensors, saved at (scene #) / (layer name) / (channel #).png
            path = os.path.join("C:/Users/User/Desktop/wooiljung/IML Projects/Feature Map visualization for CC/results/", str(index) + "/" + layer_name + "/")
            isExist = os.path.exists(path)
            if not isExist:
                os.makedirs(path)

            # change floats into 0~255 for correct picture saving
            image = np.uint8(in_tensor_np[i][j][:][:])
            # normalize option is always on, except for the first 3-channel convolution of the map (turned off)
            if normalize:
                image = image_normalize(image)

            im = Image.fromarray(image)
            im.convert("L")
            im = im.resize((240, 180), resample=Image.BOX)  # Image.BOX to prevent gradients when upscaling
            im.save(os.path.join(path, str(j)) + '.png', 'PNG')
