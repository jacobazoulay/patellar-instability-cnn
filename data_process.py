from pydicom import dcmread
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import os


# All x-rays should look like this, though perhaps flipped/rotated.You might expect a sideways x-ray but not an
# upside-down one.
# Images will have different brightnesses, contrasts, etc.


def load_data(scale_dim=512, n=None, user_os='mac'):
    home_dir = os.getcwd()
    # Read data labels Excel file, which includes image directory location and label (x, y) information
    label_dir = home_dir + '\\labels.xlsx'
    if user_os == 'mac':
        label_dir = label_dir.replace("\\", "/")
    # noinspection PyArgumentList
    labels = pd.read_excel(label_dir)
    data_labels = labels[['superior_patella_x', 'inferior_patella_x',
                          'tibial_plateau_x', 'superior_patella_y',
                          'inferior_patella_y', 'tibial_plateau_y']]
    data_labels = data_labels.to_numpy()[:n]

    if n is None:
        n = len(data_labels)
    data = []
    for i in range(n):
        print("Processing image: ", i + 1, " / ", n)

        # load and store images in data array
        image_path = home_dir + '\\Images\\' + labels.iloc[i]['lateral x-ray']
        if user_os == 'mac':
            image_path = image_path.replace("\\", "/")
        ds = dcmread(image_path)
        image = ds.pixel_array  # pixel data is stored in 'pixel_array' element which is like a np array
        data.append(image)

        # scale labels
        if scale_dim is not None:
            x_pix_dim = image.shape[1]
            y_pix_dim = image.shape[0]
            data_labels[i][:3] *= scale_dim / x_pix_dim
            data_labels[i][3:] *= scale_dim / y_pix_dim

    # scale images
    if scale_dim is not None:
        data = [cv2.resize(image, (scale_dim, scale_dim)) for image in data]

    # convert to np array
    data = np.array(data)
    data_labels = np.array(data_labels)

    return data, data_labels


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


def unscale(image, label):
    raise NotImplementedError


def save_pix_dim(data):
    y_pix_dim = [len(image) for image in data]
    x_pix_dim = [len(image[0]) for image in data]

    y_pix_dim_file = open("y_pix_dim.txt", "w")
    for element in y_pix_dim:
        y_pix_dim_file.write(str(element) + "\n")
    y_pix_dim_file.close()

    x_pix_dim_file = open("x_pix_dim.txt", "w")
    for element in x_pix_dim:
        x_pix_dim_file.write(str(element) + "\n")
    x_pix_dim_file.close()


def load_pix_dim():
    y_pix_dim = open("y_pix_dim.txt").readlines()
    x_pix_dim = open("x_pix_dim.txt").readlines()
    y_pix_dim = [int(i) for i in y_pix_dim]
    x_pix_dim = [int(i) for i in x_pix_dim]
    return x_pix_dim, y_pix_dim


data, data_labels = load_data(scale_dim=512, n=10)
print('Shape of image array: ', data.shape)
print('Shape of labels array: ', data_labels.shape)

for i in range(len(data)):
    show_image(data[i], data_labels[i])
