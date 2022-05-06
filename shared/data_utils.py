import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import count as count_func
import json
from collections import defaultdict

from PIL import Image
import io


def load_img(args, path, method=0):
    img = cv2.imread(path, method)
    #img = cv2.resize(img, (512,512))
    if method == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def cv_imshow(name, data):
    cv2.imshow(name, data)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_img_gt(args, title, data):
    fig = plt.figure()
    n_images = len(data)
    cols = 1
    for i in range(n_images):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), i + 1)
        plot = plt.imshow(data[i])
        plot.axes.get_xaxis().set_visible(False)
        plot.axes.get_yaxis().set_visible(False)
        a.set_title(title[i])
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    fig.tight_layout()
    fig.patch.set_alpha(1)
    plt.show(block=True)


def save_prediction(args, title, data, bid, img_id, epoch):
    fig = plt.figure()
    n_images = len(data)
    cols = 1
    for i in range(n_images):
        a = fig.add_subplot(cols, int(np.ceil(n_images/float(cols))), i + 1)
        plot = plt.imshow(data[i])
        plot.axes.get_xaxis().set_visible(False)
        plot.axes.get_yaxis().set_visible(False)
        a.set_title(title[i])
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    fig.tight_layout()
    fig.patch.set_alpha(1)
    #plt.show(block=False)
    plot_path = os.path.join(args.odir, 'images', str(epoch), str(bid))
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    fig.savefig(os.path.join(plot_path, title[0].split('/')[-1].replace('.png', str(img_id) + '.png')), facecolor=fig.get_facecolor())
    plt.close()


def save_cdi_imgs(data, labels, fnames, split):
    home_dir = os.getcwd()
    outdir = os.path.join(home_dir, "data/CDI", split)
    os.makedirs(outdir, exist_ok=True)

    for i in range(len(data)):
        outfile = os.path.join(outdir, fnames[i] + ".png")
        print("saving %s" % (fnames[i]))
        cv2.imwrite(outfile, data[i])


def save_cdi_labels(labels, fnames):
    home_dir = os.getcwd()
    outdir = os.path.join(home_dir, "data/CDI")
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, "labels.json")
    out = {}

    for i in range(len(labels)):
        out[fnames[i]] = labels[i]
    
    with open(outfile, 'w') as f:
        json.dump(out, f)
    
if __name__=="__main__":
    dat = np.zeros((2, 32, 32, 3))
    labels = np.zeros((2, 6)).tolist()
    fnames = ["f1", "f2"]
    #test for saving labels
    save_cdi_imgs(dat, labels, fnames, "train")
    save_cdi_labels(labels, fnames)

