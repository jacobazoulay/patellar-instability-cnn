import sys
import os
import os.path as osp

def addpath(path):
    sys.path.insert(0, path)

# add project folder and subfolders
project_folder =os.getcwd()
addpath(project_folder)
addpath(osp.join(project_folder, 'models/'))
addpath(osp.join(project_folder, 'utils/'))