import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import count as count_func
import json
import torch
from collections import defaultdict

from PIL import Image
import io

from datetime import datetime

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


def un_norm_avg_key_dist(mean_im, std_im, target, pred):
    mean_im = torch.tensor(mean_im)
    std_im = torch.tensor(std_im)
    target = target * std_im + mean_im
    pred = pred * std_im + mean_im
    d1 = torch.sqrt(torch.square(target[:, 0] - pred[:, 0]) + torch.square(target[:, 1] - pred[:, 1]))
    d2 = torch.sqrt(torch.square(target[:, 2] - pred[:, 2]) + torch.square(target[:, 3] - pred[:, 3]))
    d3 = torch.sqrt(torch.square(target[:, 4] - pred[:, 4]) + torch.square(target[:, 5] - pred[:, 5]))
    out = torch.mean((d1 + d2 + d3) / 3)
    return out


def un_standard_avg_key_dist(target, pred):
    target = (target + 1) * 64
    pred = (pred + 1) * 64
    d1 = torch.sqrt(torch.square(target[:, 0] - pred[:, 0]) + torch.square(target[:, 1] - pred[:, 1]))
    d2 = torch.sqrt(torch.square(target[:, 2] - pred[:, 2]) + torch.square(target[:, 3] - pred[:, 3]))
    d3 = torch.sqrt(torch.square(target[:, 4] - pred[:, 4]) + torch.square(target[:, 5] - pred[:, 5]))
    out = torch.mean((d1 + d2 + d3) / 3)
    return out

def compute_CDI(superior_patella_x, inferior_patella_x, tibial_plateau_x, superior_patella_y, inferior_patella_y, tibial_plateau_y):
    # x and y distances between inferior patella and tibia
    patella_to_anterior_tibia_x = tibial_plateau_x - inferior_patella_x
    patella_to_anterior_tibia_y = tibial_plateau_y - inferior_patella_y

    # x and y distances between superior and inferior patella
    patella_articular_surface_x = superior_patella_x - inferior_patella_x
    patella_articular_surface_y = superior_patella_y - inferior_patella_y

    # lengths of lines in pixels (unit doesn't matter since CDI is a ratio)
    patella_to_anterior_tibia_pixel_length = torch.sqrt(patella_to_anterior_tibia_x**2 + patella_to_anterior_tibia_y**2)
    patella_articular_surface_pixel_length = torch.sqrt(patella_articular_surface_x**2 + patella_articular_surface_y**2)

    caton_deschamps_index = patella_to_anterior_tibia_pixel_length/patella_articular_surface_pixel_length 

    return caton_deschamps_index


def show_image(image, label=None):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    plt.imshow(image, cmap='bone')

    #un-normalize images and labels before displaying
    project_dir = os.getcwd()
    label_cache_stats = json.load(open(os.path.join(project_dir, "data/CDI/cache/label_stats.json")))
    img_mean = np.load(os.path.join(project_dir, "./data/CDI/cache/im_mean.npy"))
    img_std = np.load(os.path.join(project_dir, "./data/CDI/cache/im_std.npy"))
    label_mean = label_cache_stats['label_mean']
    label_std = label_cache_stats['label_std']

    image = un_normalize(img_mean, img_std, image)

    if label is not None:
        # label = un_normalize(label_mean, label_std, label)
        label = (label + 1) * 64
        plt.scatter(label[0], label[3])  # superior patella loc in blue
        plt.scatter(label[1], label[4])  # inferior patella loc in orange
        plt.scatter(label[2], label[5])  # tibial_plateau loc in green

    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

def plot_labels(label, c):
    plt.scatter(label[0], label[3], color=c)  # superior patella loc in blue
    plt.scatter(label[1], label[4], color=c)  # inferior patella loc in orange
    plt.scatter(label[2], label[5], color=c)  # tibial_plateau loc in green


def save_prediction(args, img, imgname, gt_kpts, pred_kpts, bid, img_id, epoch):
    #unnormalize img and labels before saving
    image = un_normalize(args.img_mean, args.img_std, img)
    # labels_pred = un_normalize(args.label_mean, args.label_std, pred_kpts)
    # labels_gt = un_normalize(args.label_mean, args.label_std, gt_kpts)
    labels_pred = (pred_kpts + 1) * 64
    labels_gt = (gt_kpts + 1) * 64


    fig = plt.figure()
    plot = plt.imshow(image, cmap='bone')
    
    plot_labels(labels_gt, 'green')
    plot_labels(labels_pred, 'red')

    plot.axes.get_xaxis().set_visible(False)
    plot.axes.get_yaxis().set_visible(False)
    plt.title(imgname)
    fig.set_size_inches(np.array(fig.get_size_inches()))
    fig.tight_layout()
    fig.patch.set_alpha(1)
    #plt.show(block=False)
    
    now = datetime.now()
    dt_str = now.strftime("%m_%d_%Y %H_%M_%S")

    plot_path = os.path.join(args.odir, 'images', str(epoch), str(bid))
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    fig.savefig(os.path.join(plot_path, imgname + str(img_id) + str(dt_str) + '.png'), facecolor=fig.get_facecolor())
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

