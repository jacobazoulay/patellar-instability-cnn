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
    img = np.load(path)
    return img


def un_normalize(mean_im, std_im, *argv):
    out = []
    for arg in argv:
        un_norm = arg * std_im + mean_im
        un_norm = un_norm.astype('uint8')
        out.append(un_norm)
    if len(out) == 1:
        out = out[0]
    return out


def show_image(image, label=None):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    plt.imshow(image, cmap='gray')

    if label is not None:
        plt.scatter(label[0], label[3])  # superior patella loc in blue
        plt.scatter(label[1], label[4])  # inferior patella loc in orange
        plt.scatter(label[2], label[5])  # tibial_plateau loc in green

    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


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


def plot_labels(label, c):
    plt.scatter(label[0], label[3], color=c)  # superior patella loc in blue
    plt.scatter(label[1], label[4], color=c)  # inferior patella loc in orange
    plt.scatter(label[2], label[5], color=c)  # tibial_plateau loc in green


def save_prediction(args, img, imgname, gt_kpts, pred_kpts, bid, img_id, epoch):
    fig = plt.figure()
    plot = plt.imshow(img, cmap='gray')
    
    plot_labels(gt_kpts, 'green')
    plot_labels(pred_kpts*255.0, 'red')

    plot.axes.get_xaxis().set_visible(False)
    plot.axes.get_yaxis().set_visible(False)
    plt.title(imgname)
    fig.set_size_inches(np.array(fig.get_size_inches()))
    fig.tight_layout()
    fig.patch.set_alpha(1)
    plt.show(block=False)
    
    plot_path = os.path.join(args.odir, 'images', str(epoch), str(bid))
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    fig.savefig(os.path.join(plot_path, imgname + str(img_id) + '.png'), facecolor=fig.get_facecolor())
    plt.close()


def save_cdi_imgs(data, fnames, split):
    home_dir = os.getcwd()
    outdir = os.path.join(home_dir, "data","CDI", split)
    os.makedirs(outdir, exist_ok=True)
    print("saving Images...")
    for i in range(len(data)):
        outfile = os.path.join(outdir, fnames[i] + ".npy")
        # print("saving %s" % (fnames[i]))
        np.save(outfile, data[i])
        # cv2.imwrite(outfile, data[i])


def save_cdi_labels(labels, fnames):
    home_dir = os.getcwd()
    outdir = os.path.join(home_dir, "data", "CDI")
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, "labels.json")
    out = {}
    if os.path.exists(outfile):
        out = json.load(open(outfile))
    print("saving Labels...")
    for i in range(len(labels)):
        out[fnames[i]] = labels[i]
    
    with open(outfile, 'w') as f:
        json.dump(out, f)


def save_cdi_cache(im_stats, label_stats):
    # save normalization statistics to un-scale after fed through the model
    home_dir = os.getcwd()
    outdir = os.path.join(home_dir, "data", "CDI", "cache")
    os.makedirs(outdir, exist_ok=True)
    print("saving Images...")
    fnames = ["im_mean", "im_std"]
    for i in range(len(im_stats)):
        outfile = os.path.join(outdir, fnames[i] + ".npy")
        np.save(outfile, im_stats[i])

    outfile = os.path.join(outdir, "label_stats.json")
    out = {}
    if os.path.exists(outfile):
        out = json.load(open(outfile))
    print("saving Labels...")
    fnames = ["label_mean", "label_std"]
    for i in range(len(label_stats)):
        out[fnames[i]] = label_stats[i]

    with open(outfile, 'w') as f:
        json.dump(out, f)


if __name__ == "__main__":
    pass

