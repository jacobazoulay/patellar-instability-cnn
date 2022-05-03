# CS231N_Project
Final project for CS231N Deep Learning for Computer Vision at Stanford University
Authors: 

## Authors: 

## A. Running the code

### I. Create Virtual Environment
    $PYTHON_BIN = path/to/python/bin
    virtualenv -p $PYTHON_BIN venv
    source activate venv/bin/activate
   
### II. Install requirements
    pip install -r requirements.txt

### II. Testing the dogcat dataloader
Download the data [here](https://drive.google.com/drive/folders/1lVt1PJJ9F09MznNEZAG6yIeYBaKL1PIp?usp=sharing) and place the 'data' folder in project folder

    python shared/datasets/dogcat.py

### III. Overfitting on small dataset
    ./overfit.sh

### IV. Training/Testing a network
    1. Set parameters in run.sh
    2. Run: ./run.sh