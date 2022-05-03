import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import count as count_func
import json
from collections import defaultdict

from PIL import Image
import io


def load_img(args, path, method=1):
    img = cv2.imread(path, method)
    img = cv2.resize(img, (256,256))
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
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), i + 1)
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

def save_prediction_s3(args, title, data, bid, img_id, epoch, meta):
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

    img_data = io.BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)

    s3 = boto3.resource(service_name='s3', aws_access_key_id=args.key_id, \
                        aws_secret_access_key=args.key)
    s3bucket = s3.Bucket(args.bucket_name)

    plot_path = os.path.join(args.odir, 'images', str(epoch), str(bid))
    fname = meta.split("/")[-1].replace(".JPG", ".png")
    outfile = os.path.join(plot_path, fname)
    s3bucket.put_object(Body=img_data, ContentType='image/png', Key=outfile)


def load_data_args(args):
    args.allMats = json.loads(open('./data/CCOMATS/ccotextures.json').read())
    args.fineMats = [mat['folder'].split('/')[0].replace(' ', '') for mat in args.allMats]
    args.coarseMats = np.unique([mat['folder'].split('#')[0].replace(' ', '') for mat in args.allMats])
    lines = sorted(open('data/CCOMATS/colors1000.txt').read().splitlines())
    args.allRGB = np.asarray([[[int(round(float(c)*255))] for c in line.split(',')] for line in lines]).reshape(-1, 3)

    #create a name to index dictionary for coarse materials
    args.coarseMatsDict = defaultdict(count_func(0).__next__)
    for mat in args.coarseMats:
        args.coarseMatsDict[mat]

    args.fineDict = {}
    args.coarseDict = {}
    for idx in range(len(args.fineMats)):
        args.fineDict[str(tuple(args.allRGB[idx]))] = idx
        matname = args.fineMats[idx].split('#')[0]
        args.coarseDict[str(tuple(args.allRGB[idx]))] = args.coarseMatsDict[matname]
    args.num_classes = len(args.fineMats) if args.fg else len(args.coarseMats)
    args.mats = args.fineMats if args.fg else args.coarseMats