# --------------------------------------------------------
#Copyright 2021, Micael Tchapmi, All rights reserved
# --------------------------------------------------------
import random
import os, sys
import argparse
import numpy as np
import cv2
import glob

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_process import DataProcess, get_while_running, kill_data_processes
from data_utils import load_img, show_img_gt


EXTS = ['.png', '.jpeg', '.jpg']


class DogCatDataProcess(DataProcess):

    def __init__(self, data_queue, args, split='train', repeat=True):
        """DogCat dataloader.
        Args:
            data_queue: multiprocessing queue where data is stored at.
            split: str in ('train', 'val', 'test'). Loads corresponding dataset.
            repeat: repeats epoch if true. Terminates after one epoch otherwise.
        """
        args.DATA_PATH = "data/%s/%s" % (args.dataset, split)
        
        args.num_classes = 2
        args.labels = {'dog': 0, 'cat': 1}
        args.idx2label = {0: 'dog', 1: 'cat'}
        data_paths = [f for f in glob.glob(args.DATA_PATH+"/*")]

        self.data_paths = data_paths
        self.args = args
        random.shuffle(self.data_paths)
        super().__init__(data_queue, self.data_paths, None, args.batch_size, repeat=repeat)

    def getSample(self, args, fname):
        #load imgs and append labels
        imgs = load_img(args, fname, 1)
        imgname = fname.split('/')[-1]
        category = imgname.split('.')[0]
        if category not in args.labels:
            raise Exception("invalid label")
        label = args.labels[category]
        gts = np.asarray(label)
        meta = [fname, category, label]

        return imgs, gts, meta

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
    args.dataset = 'DogCat' #dataset name e.g DogCat
    args.nworkers = 1
    args.batch_size = 2
    data_processes = []
    data_queue = Queue(8)
    for i in range(args.nworkers):
        data_processes.append(DogCatDataProcess(data_queue, args, split='train',
                                               repeat=False))
        data_processes[-1].start()
    N = len(data_processes[0].data_paths)
    batch_size = data_processes[0].batch_size
    Nb = int(N/batch_size)
    if Nb*batch_size < N:
        Nb += 1

    for imgs, gts, meta in get_while_running(data_processes, data_queue, 0):
        #check labels visually
        n, w, h, c = imgs.shape
        for i in range(len(imgs)):
            (fname, category, label_id) = meta[i]
            cv2.imshow(category,imgs[i])
            cv2.waitKey(0)
        break
    kill_data_processes(data_queue, data_processes)


if __name__ == '__main__':
    test_process()