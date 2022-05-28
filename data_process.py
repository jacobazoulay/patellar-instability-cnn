from pydicom import dcmread
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import os
import json
import platform
import albumentations as A
import random
from shared.data_utils import save_cdi_imgs, save_cdi_labels, save_cdi_cache


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
        if platform.system() == 'Darwin' or platform.system() == 'Linux':
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
    if platform.system() == 'Darwin' or platform.system() == 'Linux':
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
                         keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
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


def rescale_images(data, data_labels, scale_dim=128):
    scaled_images = []
    scaled_labels = []
    # reshape data_labels to tuples for transform operation
    keypoints = list(zip(zip(data_labels[:, 0], data_labels[:, 3]),
                         zip(data_labels[:, 1], data_labels[:, 4]),
                         zip(data_labels[:, 2], data_labels[:, 5])))

    # crop image and labels into squares
    for i in range(len(data)):
        scale = A.Compose([A.Resize(width=scale_dim, height=scale_dim)],
                          keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
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

    train_img_name = np.array([full_labels.iloc[i]['lateral x-ray'].replace("\\", "_") for i in train_idx])
    val_img_name = np.array([full_labels.iloc[i]['lateral x-ray'].replace("\\", "_") for i in val_idx])
    test_img_name = np.array([full_labels.iloc[i]['lateral x-ray'].replace("\\", "_") for i in test_idx])

    train_data, train_data_labels = data[train_idx, :, :], data_labels[train_idx, :]
    val_data, val_data_labels = data[val_idx, :, :], data_labels[val_idx, :]
    test_data, test_data_labels = data[test_idx, :, :], data_labels[test_idx, :]

    return train_data, train_data_labels, train_img_name, \
           val_data, val_data_labels, val_img_name, \
           test_data, test_data_labels, test_img_name


def show_image(image, label=None, title=None):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    plt.imshow(image, cmap='bone')

    if title is not None:
        plt.title(title)

    #un-normalize images and labels before displaying
    
    project_dir = os.getcwd()
    img_mean = np.load(os.path.join(project_dir, "./data/CDI/cache/im_mean.npy"))
    img_std = np.load(os.path.join(project_dir, "./data/CDI/cache/im_std.npy"))
    label_cache_stats = json.load(open(os.path.join(project_dir, "data/CDI/cache/label_stats.json")))
    label_mean = label_cache_stats['label_mean']
    label_std = label_cache_stats['label_std']
    
    image = un_normalize(img_mean, img_std, image)

    if label is not None:
        label = un_normalize(label_mean, label_std, label)
        plt.scatter(label[0], label[3])  # superior patella loc in blue
        plt.scatter(label[1], label[4])  # inferior patella loc in orange
        plt.scatter(label[2], label[5])  # tibial_plateau loc in green

    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


def show_image_no_norm(image, label=None):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    plt.imshow(image, cmap='bone')

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
        A.Rotate(limit=(-45, 45), p=1, border_mode=4),
        A.InvertImg(p=0.2),
        #A.VerticalFlip(p=0.3),
        #A.HorizontalFlip(p=0.3)
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    idxs = np.random.randint(0, len(data), size=n)

    # reshape data_labels to tuples for transform operation
    keypoints = list(zip(zip(data_labels[:, 0], data_labels[:, 3]),
                         zip(data_labels[:, 1], data_labels[:, 4]),
                         zip(data_labels[:, 2], data_labels[:, 5])))

    aug_images, aug_labels, aug_names = [], [], []

    balanceLRFlag = False # Flag to determine if balance left and right images - set false to test specifically left or right images

    if balanceLRFlag:
        #sort the data into left and right images to balance the data
        left_data, left_keypoints, left_data_names = [], [], []
        right_data, right_keypoints, right_data_names = [], [], []
        for idx, name in enumerate(data_names):
            direction = name.split("_")[0][len("JUPITER") + 5 - 1] # L or R
            if direction not in ['R', 'L']:
                raise Exception("invalid character %s for x-ray direction" % (direction))
            if direction == 'R':
                right_data.append(data[idx])
                right_keypoints.append(keypoints[idx])
                right_data_names.append(data_names[idx])
            elif direction == 'L':
                left_data.append(data[idx])
                left_keypoints.append(keypoints[idx])
                left_data_names.append(data_names[idx])
        n_left = len(left_data)
        n_right = len(right_data)
        assert n_left > 0, "need at least one left image"
        assert n_right > 0, "need at least one right image"
        assert len(left_data) == len(left_keypoints) == len(left_data_names), "unequal left data and label lengths"
        assert len(right_data) == len(right_keypoints) == len(right_data_names), "unequal right data and label lengths"
        print("length of left training data before augmentation: %d" % (n_left))
        print("length of right training data before augmentation: %d" % (n_right))                     

        left_idx = 0
        right_idx = 0
        for i in range(len(idxs)):
            if n_left > n_right: # augment a right image
                cur_data = right_data[right_idx]
                cur_keypoints = right_keypoints[right_idx]
                cur_name = right_data_names[right_idx]
                right_idx = (right_idx + 1) % len(right_data)
                n_right += 1
            elif n_left <= n_right: # augment a left image
                cur_data = left_data[left_idx]
                cur_keypoints = left_keypoints[left_idx]
                cur_name = left_data_names[left_idx]
                left_idx = (left_idx + 1) % len(left_data)
                n_left += 1

            #augmented = augment(image=data[idxs[i]], keypoints=keypoints[idxs[i]])
            augmented = augment(image=cur_data, keypoints=cur_keypoints)
            augmented_image = augmented['image']
            augmented_keypoints = augmented['keypoints']

            # reshape keypoint tuples back to original data_label shape
            augmented_keypoints = [augmented_keypoints[0][0], augmented_keypoints[1][0],
                                augmented_keypoints[2][0], augmented_keypoints[0][1],
                                augmented_keypoints[1][1], augmented_keypoints[2][1]]

            #augmented_name = data_names[idxs[i]][:-4] + "_aug" + str((idxs[:i + 1] == idxs[i]).sum()) + ".dcm"
            augmented_name = cur_name[:-4] + "_aug" + str(i) + ".dcm"

            aug_images.append(augmented_image)
            aug_labels.append(augmented_keypoints)
            aug_names.append(augmented_name)

        print("length of left training data after augmentation: %d" % (n_left))
        print("length of right training data after augmentation: %d" % (n_right))
    else: # Don't balance left and right
        for i in idxs:
            augmented = augment(image = data[i], keypoints = keypoints[i])
            augmented_image = augmented['image']
            augmented_keypoints = augmented['keypoints']

            augmented_keypoints = [augmented_keypoints[0][0], augmented_keypoints[1][0],
                                augmented_keypoints[2][0], augmented_keypoints[0][1],
                                augmented_keypoints[1][1], augmented_keypoints[2][1]]

            #augmented_name = data_names[idxs[i]][:-4] + "_aug" + str((idxs[:i + 1] == idxs[i]).sum()) + ".dcm"
            augmented_name = cur_name[:-4] + "_aug" + str(i) + ".dcm"

            aug_images.append(augmented_image)
            aug_labels.append(augmented_keypoints)
            aug_names.append(augmented_name)

    # convert to np array
    aug_images = np.array(aug_images)
    aug_labels = np.array(aug_labels)
    aug_names = np.array(aug_names)

    print(data_names)
    print(aug_names)

    return aug_images, aug_labels, aug_names


def sub_mean(data, *argv):
    data = data.astype('float64')
    mean_im = np.mean(data, axis=0)
    std_im = np.std(data, axis=0)
    data = (data - mean_im) / (std_im + 1e-10)
    out = [data]
    for arg in argv:
        norm = (arg - mean_im) / (std_im + 1e-10)
        out.append(norm)
    out.extend([mean_im, std_im])
    return out


def norm_labels(*argv):
    out = []
    for arg in argv:
        norm = arg / 64 - 1
        out.append(norm)
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


def calibrate_canny():
    # used to visualize which canny edge detection threshold parameters work best for this data
    # use sliders to control bounds
    # press 'Esc' key to exit

    # load data (images and labels), center crop data, and down-scale data
    data, data_labels = load_data(n=1)
    data, data_labels = crop_images(data, data_labels)
    data, data_labels = rescale_images(data, data_labels, scale_dim=128)

    def callback(x):
        print(x)

    img = data[0]  # read image as grayscale

    canny = cv2.Canny(img, 85, 255)

    cv2.namedWindow('image')  # make a window with name 'image'
    cv2.createTrackbar('Lower', 'image', 0, 255, callback)  # lower threshold trackbar for window 'image
    cv2.createTrackbar('Upper', 'image', 0, 255, callback)  # upper threshold trackbar for window 'image

    while (1):
        numpy_horizontal_concat = np.concatenate((img, canny), axis=1)  # to display image side by side
        cv2.imshow('image', numpy_horizontal_concat)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # escape key
            break
        l = cv2.getTrackbarPos('Lower', 'image')
        u = cv2.getTrackbarPos('Upper', 'image')

        canny = cv2.Canny(img, l, u)

    cv2.destroyAllWindows()


def main():
    # load data (images and labels), center crop data, and down-scale data
    n_imgs = None  # None loads all 304 images
    n_aug = 5000   # number of augmented images to create
    use_edges = False   # whether to use edge detection transformations

    data, data_labels = load_data(n=n_imgs)
    data, data_labels = crop_images(data, data_labels)
    data, data_labels = rescale_images(data, data_labels, scale_dim=128)
    print('\nShape of image array after crop and rescale: ', data.shape)
    print('Shape of labels array after crop and rescale: ', data_labels.shape, '\n')

    # split data in train, validation, and test sets
    train_data, train_data_labels, train_data_names, \
    val_data, val_data_labels, val_data_names, \
    test_data, test_data_labels, test_data_names = train_val_test_split(data, data_labels)
    print('Train / Validation / Test:  ', train_data.shape[0], ' / ', val_data.shape[0], ' / ', test_data.shape[0])

    # augment training data
    train_aug, train_aug_labels, train_aug_names = augment_data(train_data, train_data_labels, train_data_names, n=n_aug)
    print('Augment size: ', train_aug.shape[0])

    # add augmented data to training set
    train_data = np.append(train_data, train_aug, axis=0)
    train_data_labels = np.append(train_data_labels, train_aug_labels, axis=0)
    train_data_names = np.append(train_data_names, train_aug_names, axis=0)
    print('Train + augmented size: ', train_data.shape[0], '\n')

    if use_edges:
        train_data_edge, val_data_edge, test_data_edge = [], [], []
        for im in train_data:
            edge = np.array(cv2.Canny(im, 30, 188))
            train_data_edge.append(edge)
        for im in val_data:
            edge = np.array(cv2.Canny(im, 30, 188))
            val_data_edge.append(edge)
        for im in test_data:
            edge = np.array(cv2.Canny(im, 30, 188))
            test_data_edge.append(edge)

        # convert to numpy array
        train_data_edge = np.array(train_data_edge)
        val_data_edge = np.array(val_data_edge)
        test_data_edge = np.array(test_data_edge)

        # normalize edge images
        train_data_edge, val_data_edge, test_data_edge, mean_im_edge, std_im_edge = sub_mean(train_data_edge, val_data_edge, test_data_edge)
        train_data_labels, val_data_labels, test_data_labels, mean_label, std_label = sub_mean(train_data_labels, val_data_labels, test_data_labels)

        # save edge images to local directory
        save_cdi_imgs(train_data_edge, train_data_names, "train")
        save_cdi_imgs(val_data_edge, val_data_names, "val")
        save_cdi_imgs(test_data_edge, test_data_names, "test")

        # save original images to local directory for reference and visualization
        save_cdi_imgs(train_data, train_data_names, "train_orig")
        save_cdi_imgs(val_data, val_data_names, "val_orig")
        save_cdi_imgs(test_data, test_data_names, "test_orig")

        # save edge image and label normalization stats for unscaling
        save_cdi_cache([mean_im_edge, std_im_edge], [list(mean_label), list(std_label)])

    else:
        # normalize the images using training set statistics
        train_data, val_data, test_data, mean_im, std_im = sub_mean(train_data, val_data, test_data)
        # train_data_labels, val_data_labels, test_data_labels, mean_label, std_label = norm_labels(train_data_labels, val_data_labels, test_data_labels)
        train_data_labels, val_data_labels, test_data_labels = norm_labels(train_data_labels, val_data_labels,
                                                                           test_data_labels)
        mean_label = np.zeros(6)
        std_label = np.ones(6) / 64
        # save images to local directory
        save_cdi_imgs(train_data, train_data_names, "train")
        save_cdi_imgs(val_data, val_data_names, "val")
        save_cdi_imgs(test_data, test_data_names, "test")

        # save image and label normalization stats for unscaling
        save_cdi_cache([mean_im, std_im], [list(mean_label), list(std_label)])

    # save labels to local directory
    save_cdi_labels(train_data_labels.tolist(), train_data_names)
    save_cdi_labels(val_data_labels.tolist(), val_data_names)
    save_cdi_labels(test_data_labels.tolist(), test_data_names)

if __name__ == "__main__":
    main()