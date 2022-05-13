# CS231N_Project
Final project for CS231N Deep Learning for Computer Vision at Stanford University
Authors: 

## Authors: 

## A. Running the code

## Data Preprocessing
    python data_process.py

## Clone Repository
    git clone git@github.com:jacobazoulay/CS231N_Project.git

### Create Virtual Environment
    $PYTHON_BIN = path/to/python/bin
    virtualenv -p $PYTHON_BIN venv
    source activate venv/bin/activate
   
### Install requirements
    pip install -r requirements.txt

### Training/Testing a network
    1. Set parameters in run.sh
    2. Run: ./run.sh
    3. Results are saved in './results' directory
    
### Plotting loss after training
    python tools/plot_loss.py path/to/results/trainlog.txt

### Visualizing Predictions on Images
    1. Run evaluation by setting train=0 in run.sh
    2. Predictions are stored in path/to/results/Images