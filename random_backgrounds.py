import os
from scipy import misc
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import requests
from io import BytesIO

root_path = '/home/paulo/MockDataset1/'
image_net_path = '/home/paulo/imagenet_50k.txt'

img_res = (480, 640)

# read imagenet
with open(image_net_path) as f:
    content = f.readlines()

img_randomized_idxs = list(range(len(content)))
np.random.shuffle(img_randomized_idxs)

def get_random_imagenet_img(idx):
    image_url = content[img_randomized_idxs[idx]].split('\t')[1].strip()
    response = requests.get(image_url)
    imagenet_img = Image.open(BytesIO(response.content))
    imagenet_img = np.array(imagenet_img)
    imagenet_img = misc.imresize(imagenet_img, img_res)
    return imagenet_img

def try_to_get_random_imagenet_img(idx, plot=False):
    max_attempts = 100
    img_not_succeded = True
    imagenet_img = None
    attempt_num = 0
    while img_not_succeded:
        try:
            imagenet_img = get_random_imagenet_img(idx)
            if plot:
                plt.imshow(imagenet_img)
                plt.show()
            img_not_succeded = False
        except:
            idx += 1
            img_not_succeded = True
        attempt_num += 1
        if attempt_num > max_attempts:
            return None, -1
    return imagenet_img, idx

idx = 0
for root, dirs, files in os.walk(root_path, topdown=True):
    for filename in sorted(files):
        curr_file_ext = filename.split('.')[1]
        if curr_file_ext == 'png' and not 'depth' in filename:
            imagenet_img, idx = try_to_get_random_imagenet_img(idx)

            img = misc.imread(root + '/' + filename)
            img2 = np.logical_and(img[:, :, 1] > 244, img[:, :, 0] < 10)
            img[img2, 0:3] = imagenet_img[img2, :]

            plt.imshow(img)
            plt.title(root + '/' + filename)
            plt.show()

            idx += 1
