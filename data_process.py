from pydicom import dcmread
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import os
import platform

# All x-rays should look like this, though perhaps flipped/rotated.You might expect a sideways x-ray but not an
# upside-down one.
# Images will have different brightnesses, contrasts, etc.


def load_data(scale_dim=512, n=None, crop=True, subtract_mean=False):
    home_dir = os.getcwd()
    data_labels, full_labels = load_data_labels()   # full_labels includes image directory information
    data_labels = data_labels[:n]

    if n is None:
        n = len(data_labels)
    data = []
    for i in range(n):
        print("Processing image: ", i + 1, " / ", n)

        # load and store images in data array
        image_path = home_dir + '\\Images\\' + full_labels.iloc[i]['lateral x-ray']
        if platform.system() == 'Darwin':
            image_path = image_path.replace("\\", "/")
        ds = dcmread(image_path)
        image = ds.pixel_array  # pixel data is stored in 'pixel_array' element which is like a np array
        data.append(image)

    # store pixel dimension in cache for scaling back
    x_pix_dim = [len(image[0]) for image in data]
    y_pix_dim = [len(image) for image in data]
    data_cache = [x_pix_dim, y_pix_dim]

    # crop images
    if crop:
        data, data_labels = crop_images(data, data_labels)

    # scale images
    if scale_dim is not None:
        data, data_labels = rescale_images(data, data_labels, scale_dim)

    # subtract mean
    if subtract_mean:
        data = sub_mean(data)

    # convert to np array
    data = np.array(data).astype('float64')
    data_labels = np.array(data_labels)
    data_cache = np.array(data_cache)

    return data, data_labels, data_cache


def load_data_labels():
    home_dir = os.getcwd()
    # Read data labels Excel file, which includes image directory location and label (x, y) information
    label_dir = home_dir + '\\labels.xlsx'
    if platform.system() == 'Darwin':
        label_dir = label_dir.replace("\\", "/")
    # noinspection PyArgumentList
    full_labels = pd.read_excel(label_dir)
    data_labels = full_labels[['superior_patella_x', 'inferior_patella_x',
                          'tibial_plateau_x', 'superior_patella_y',
                          'inferior_patella_y', 'tibial_plateau_y']]
    data_labels = data_labels.to_numpy()
    return data_labels, full_labels


def crop_images(data, data_labels):
    cropped = []
    # crop image and labels into squares
    for i in range(len(data)):
        y_dim, x_dim = data[i].shape
        if y_dim > x_dim:
            start = (y_dim - x_dim) // 2
            end = (y_dim + x_dim) // 2
            im_cropped = data[i][start:end, :]

            data_labels[i][3:] -= start

        else:
            start = (y_dim - x_dim) // 2
            end = (y_dim + x_dim) // 2
            im_cropped = data[i][:, start:end]

            data_labels[i][:3] -= start

        cropped.append(im_cropped)
    return cropped, data_labels


def rescale_images(data, data_labels, scale_dim):
    scaled = []
    # scale labels
    for i in range(len(data)):
        scaled_im = cv2.resize(data[i], (scale_dim, scale_dim))
        y_pix_dim, x_pix_dim = data[i].shape
        data_labels[i][:3] *= scale_dim / x_pix_dim
        data_labels[i][3:] *= scale_dim / y_pix_dim
        scaled.append(scaled_im)
    return scaled, data_labels


def sub_mean(data):
    data = np.array(data).astype('float64')
    mean_im = np.mean(data, axis=0)
    std_im = np.std(data, axis=0)
    data = (data - mean_im) / std_im
    return data


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


def unscale(image, label, data_cache):
    raise NotImplementedError


# load data (images and labels) and crop, rescale, and normalize
data, data_labels, data_cache = load_data(scale_dim=512, n=10, crop=True, subtract_mean=True)
print('Shape of image array: ', data.shape)
print('Shape of labels array: ', data_labels.shape)

# show all images in data
for i in range(len(data)):
    show_image(data[i], data_labels[i])

# show mean image
mean_im = np.mean(data, axis=0)
show_image(mean_im)