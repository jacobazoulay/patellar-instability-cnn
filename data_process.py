from pydicom import dcmread
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import os
import platform
import albumentations as A
import random
from shared.data_utils import save_cdi_imgs, save_cdi_labels


def load_data(n=None):
    """
    Reads and loads raw image data using labels Excel
    :param
        n: number of images to load. If None then all images will load
    :return
        data [list]: 2-D images with pixel values scaled from [0-255]
        data_labels [np array]: corresponding data labels (x1, x2, x3, y1, y2, y3)
    """
    home_dir = os.getcwd()
    data_labels, full_labels = load_data_labels()  # full_labels includes image directory information
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
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)  # normalize pixels range [0, 255]
        data.append(image)

    return data, data_labels


def load_data_labels():
    """
    Loads image location and labels from Excel file
    :return
         data_labels (np array): data labels corresponding to images (x1, x2, x3, y1, y2, y3)
         full_labels (np array): image directory location + data_labels
    """
    home_dir = os.getcwd()
    # Read data labels Excel file, which includes image directory location and label (x, y) information
    label_dir = home_dir + '\\labels.xlsx'
    if platform.system() == 'Darwin':
        label_dir = label_dir.replace("\\", "/")

    full_labels = pd.read_excel(label_dir)
    data_labels = full_labels[['superior_patella_x', 'inferior_patella_x',
                               'tibial_plateau_x', 'superior_patella_y',
                               'inferior_patella_y', 'tibial_plateau_y']]
    data_labels = data_labels.to_numpy()

    return data_labels, full_labels


def crop_images(data, data_labels):
    cropped_images = []
    cropped_labels = []
    # reshape data_labels to tuples for transform operation
    keypoints = list(zip(zip(data_labels[:, 0], data_labels[:, 3]),
                         zip(data_labels[:, 1], data_labels[:, 4]),
                         zip(data_labels[:, 2], data_labels[:, 5])))

    # crop image and labels into squares
    for i in range(len(data)):
        dim = min(data[i].shape)
        crop = A.Compose([A.CenterCrop(width=dim, height=dim, p=1.0)],
                         keypoint_params=A.KeypointParams(format='xy'))
        cropped = crop(image=data[i], keypoints=keypoints[i])
        cropped_image = cropped['image']
        cropped_keypoints = cropped['keypoints']
        # reshape keypoint tuples back to original data_label shape
        cropped_keypoints = [cropped_keypoints[0][0], cropped_keypoints[1][0],
                             cropped_keypoints[2][0], cropped_keypoints[0][1],
                             cropped_keypoints[1][1], cropped_keypoints[2][1]]
        cropped_images.append(cropped_image)
        cropped_labels.append(cropped_keypoints)

    # convert to np array
    cropped_labels = np.array(cropped_labels)

    return cropped_images, cropped_labels


def rescale_images(data, data_labels, scale_dim=124):
    scaled_images = []
    scaled_labels = []
    # reshape data_labels to tuples for transform operation
    keypoints = list(zip(zip(data_labels[:, 0], data_labels[:, 3]),
                         zip(data_labels[:, 1], data_labels[:, 4]),
                         zip(data_labels[:, 2], data_labels[:, 5])))

    # crop image and labels into squares
    for i in range(len(data)):
        scale = A.Compose([A.Resize(width=scale_dim, height=scale_dim)],
                          keypoint_params=A.KeypointParams(format='xy'))
        scaled = scale(image=data[i], keypoints=keypoints[i])
        scaled_image = scaled['image']
        scaled_keypoints = scaled['keypoints']
        # reshape keypoint tuples back to original data_label shape
        scaled_keypoints = [scaled_keypoints[0][0], scaled_keypoints[1][0],
                            scaled_keypoints[2][0], scaled_keypoints[0][1],
                            scaled_keypoints[1][1], scaled_keypoints[2][1]]
        scaled_images.append(scaled_image)
        scaled_labels.append(scaled_keypoints)

    # convert to np array
    scaled_images = np.array(scaled_images)
    scaled_labels = np.array(scaled_labels)

    return scaled_images, scaled_labels


def train_val_test_split(data, data_labels):
    _, full_labels = load_data_labels()  # full_labels includes image directory information | data labels is array of 6 output prediction coordinates

    n = len(data)  # number of images total
    data_labels = data_labels[:n]

    train_idx, val_idx, test_idx = [], [], []

    training_n, val_n, test_n = int(n * 0.6), int(n * 0.2), int(n * 0.2)

    random.seed(1)  # initialize random seed to generate repeatable results
    processed_set = set()

    # debugging sets (can be removed)
    train_set, val_set, test_set = set(), set(), set()

    for i in range(n):
        ran_num = random.random()  # initialize random number
        img_id = full_labels.iloc[i]['lateral x-ray'][0:12]  # unique image ID (i.e. JUPITERW001R)

        if img_id not in processed_set:
            processed_set.add(img_id)

            if ran_num < 0.34 and len(test_idx) < test_n:  # add to test set
                test_idx.append(i)
                test_set.add(img_id)

                for j in range(i + 1, n):  # find all other images with same ID
                    next_img_idx = full_labels.iloc[j]['lateral x-ray'][0:12]
                    if next_img_idx == img_id:
                        test_idx.append(j)
            elif ran_num < 0.68 and len(val_idx) < val_n:  # add to val set
                val_idx.append(i)
                val_set.add(img_id)

                for j in range(i + 1, n):  # find all other images with same ID
                    next_img_idx = full_labels.iloc[j]['lateral x-ray'][0:12]
                    if next_img_idx == img_id:
                        val_idx.append(j)
            else:  # add to train set
                train_idx.append(i)
                train_set.add(img_id)

                for j in range(i + 1, n):  # find all other images with same ID
                    next_img_idx = full_labels.iloc[j]['lateral x-ray'][0:12]
                    if next_img_idx == img_id:
                        train_idx.append(j)

    train_img_name = np.array([full_labels.iloc[i]['lateral x-ray'] for i in train_idx])
    val_img_name = np.array([full_labels.iloc[i]['lateral x-ray'] for i in val_idx])
    test_img_name = np.array([full_labels.iloc[i]['lateral x-ray'] for i in test_idx])

    train_data, train_data_labels = data[train_idx, :, :], data_labels[train_idx, :]
    val_data, val_data_labels = data[val_idx, :, :], data_labels[val_idx, :]
    test_data, test_data_labels = data[test_idx, :, :], data_labels[test_idx, :]

    return train_data, train_data_labels, train_img_name, \
           val_data, val_data_labels, val_img_name, \
           test_data, test_data_labels, test_img_name


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


def augment_data(data, data_labels, data_names, n=100):
    augment = A.Compose([
        A.RandomResizedCrop(width=data.shape[1], height=data.shape[2], scale=(0.8, 1.0), ratio=(0.75, 1.33), p=0.7),
        A.RandomBrightnessContrast(p=0.9),
        A.Rotate(p=0.2),
        A.InvertImg(p=0.2),
        A.VerticalFlip(p=0.3),
        A.HorizontalFlip(p=0.3)
    ], keypoint_params=A.KeypointParams(format='xy'))

    idxs = np.random.randint(0, len(data), size=n)

    # reshape data_labels to tuples for transform operation
    keypoints = list(zip(zip(data_labels[:, 0], data_labels[:, 3]),
                         zip(data_labels[:, 1], data_labels[:, 4]),
                         zip(data_labels[:, 2], data_labels[:, 5])))

    aug_images, aug_labels, aug_names = [], [], []

    for i in range(len(idxs)):
        augmented = augment(image=data[idxs[i]], keypoints=keypoints[idxs[i]])
        augmented_image = augmented['image']
        augmented_keypoints = augmented['keypoints']

        # reshape keypoint tuples back to original data_label shape
        augmented_keypoints = [augmented_keypoints[0][0], augmented_keypoints[1][0],
                               augmented_keypoints[2][0], augmented_keypoints[0][1],
                               augmented_keypoints[1][1], augmented_keypoints[2][1]]

        augmented_name = data_names[idxs[i]][:-4] + "_aug" + str((idxs[:i + 1] == idxs[i]).sum()) + ".dcm"

        aug_images.append(augmented_image)
        aug_labels.append(augmented_keypoints)
        aug_names.append(augmented_name)

    # convert to np array
    aug_images = np.array(aug_images)
    aug_labels = np.array(aug_labels)
    aug_names = np.array(aug_names)

    return aug_images, aug_labels, aug_names


def sub_mean(data, *argv):
    data = data.astype('float64')
    mean_im = np.mean(data, axis=0)
    std_im = np.std(data, axis=0)
    data = (data - mean_im) / std_im
    out = [data]
    for arg in argv:
        norm = (arg - mean_im) / std_im
        out.append(norm)
    out.extend([mean_im, std_im])
    return out


def un_normalize(mean_im, std_im, *argv):
    out = []
    for arg in argv:
        un_norm = arg * std_im + mean_im
        un_norm = un_norm.astype('uint8')
        out.append(un_norm)
    if len(out) == 1:
        out = out[0]
    return out


def main():
    # load data (images and labels), center crop data, and down-scale data
    data, data_labels = load_data(n=None)
    data, data_labels = crop_images(data, data_labels)
    data, data_labels = rescale_images(data, data_labels, scale_dim=124)
    print('\nShape of image array after crop and rescale: ', data.shape)
    print('Shape of labels array after crop and rescale: ', data_labels.shape, '\n')

    # split data in train, validation, and test sets
    train_data, train_data_labels, train_data_names, \
    val_data, val_data_labels, val_data_names, \
    test_data, test_data_labels, test_data_names = train_val_test_split(data, data_labels)
    print('Train / Validation / Test:  ', train_data.shape[0], ' / ', val_data.shape[0], ' / ', test_data.shape[0])

    # augment training data
    train_aug, train_aug_labels, train_aug_names = augment_data(train_data, train_data_labels, train_data_names, n=400)
    print('Augment size: ', train_aug.shape[0])

    # add augmented data to training set
    train_data = np.append(train_data, train_aug, axis=0)
    train_data_labels = np.append(train_data_labels, train_aug_labels, axis=0)
    train_data_names = np.append(train_data_names, train_aug_names, axis=0)
    print('Train + augmented size: ', train_data.shape[0], '\n')

    # normalize the images using training set statistics
    train_data, val_data, test_data, mean_im, std_im = sub_mean(train_data, val_data, test_data)

    # save images to local directory
    save_cdi_imgs(train_data, train_data_names, "train")
    save_cdi_imgs(val_data, val_data_names, "val")
    save_cdi_imgs(test_data, test_data_names, "test")

    # save labels to local directory
    save_cdi_labels(train_data_labels.tolist(), train_data_names)
    save_cdi_labels(val_data_labels.tolist(), val_data_names)
    save_cdi_labels(test_data_labels.tolist(), test_data_names)


if __name__ == "__main__":
    main()
