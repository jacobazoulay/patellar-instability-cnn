# --------------------------------------------------------
#Copyright 2021, Micael Tchapmi, All rights reserved
# --------------------------------------------------------
import random
import os, sys
import argparse
import numpy as np
import cv2
import glob
import json

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataprocess import DataProcess, get_while_running, kill_data_processes
from data_utils import load_img
sys.path.insert(0, './')
from data_process import show_image


EXTS = ['.png', '.jpeg', '.jpg', '.npy']


class CDIDataProcess(DataProcess):

    def __init__(self, data_queue, args, split='train', repeat=True):
        """CDI dataloader.
        Args:
            data_queue: multiprocessing queue where data is stored at.
            split: str in ('train', 'val', 'test'). Loads corresponding dataset.
            repeat: repeats epoch if true. Terminates after one epoch otherwise.
        """
        args.DATA_PATH = "data/%s/%s" % (args.dataset, split)
        
        args.num_classes = 6
        label_names = ['superior_patella_x', 'inferior_patella_x',
                        'tibial_plateau_x', 'superior_patella_y',
                        'inferior_patella_y', 'tibial_plateau_y']
        args.labels = {}
        args.idx2label = {}
        for i, l in enumerate(label_names):
            args.labels[l] = i
            args.idx2label[i] = l
        args.label_names = label_names
        
        data_paths = [f for f in glob.glob(args.DATA_PATH+"/*")]
        labels_json_pth = "data/%s/%s" % ("CDI", "labels.json")
        labels_json = json.load(open(labels_json_pth))

        #load normalization cache
        project_dir = os.getcwd()
        label_cache_stats = json.load(open(os.path.join(project_dir, "data/CDI/cache/label_stats.json")))
        args.img_mean = np.load(os.path.join(project_dir, "./data/CDI/cache/im_mean.npy"))
        args.img_std = np.load(os.path.join(project_dir, "./data/CDI/cache/im_std.npy"))
        args.label_mean = label_cache_stats['label_mean']
        args.label_std = label_cache_stats['label_std']

        self.data_paths = data_paths
        self.labels_json = labels_json
        self.args = args
        random.shuffle(self.data_paths)
        super().__init__(data_queue, self.data_paths, None, args.batch_size, repeat=repeat)

    def getSample(self, args, fname):
        #load single img and append labels
        img = load_img(args, fname, 0)
        imgname = fname.split('/')[-1]
        basename = imgname[:-4]
        gt = np.array(self.labels_json[basename])
        meta = [fname, basename] #any extra information about image can be added here

        return img, gt, meta

    def load_data(self, fname):
        imgs, gts, meta = self.getSample(self.args, fname)
        return imgs[np.newaxis, ...], gts[np.newaxis, ...], meta

    def collate(self, batch):
        imgs, gts, meta = list(zip(*batch))
        if len(imgs) > 0:
            imgs = np.concatenate(imgs, 0)
            gts = np.concatenate(gts, 0)
        return imgs, gts, meta

def test_process():
    from multiprocessing import Queue
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()
    args.dataset = 'CDI' #dataset name
    args.nworkers = 1
    args.batch_size = 2
    data_processes = []
    data_queue = Queue(8)
    for i in range(args.nworkers):
        data_processes.append(CDIDataProcess(data_queue, args, split='train',
                                               repeat=False))
        data_processes[-1].start()
    N = len(data_processes[0].data_paths)
    batch_size = data_processes[0].batch_size
    Nb = int(N/batch_size)
    if Nb*batch_size < N:
        Nb += 1

    for imgs, gts, meta in get_while_running(data_processes, data_queue, 0):
        #check labels visually
        n, w, h = imgs.shape
        for i in range(len(imgs)):
            show_image(imgs[i], gts[i])
            #imgname = meta[i][1]
            #cv2.imshow(imgname,imgs[i])
            #cv2.waitKey(0)
        break
    kill_data_processes(data_queue, data_processes)


if __name__ == '__main__':
    test_process()