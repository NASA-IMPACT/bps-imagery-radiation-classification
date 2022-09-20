N_COLS = 325  # Number of columns in the largest image of the dataset
N_ROWS = 325  # Number of rows in the largest image of the dataset

DATA_BASE_PATH = "/path/to/data/"
GRADCAM_PATH = "./grad_cam"

DOESDICT = {
    'P278': 0,
    'P279': 0.1,
    'P280': 1,
    'P282': 0,
    'P283': 0.1,
    'P284': 1,
    'P286': 0,
    'P287': 0.1,
    'P288': 1,
    'P242': 0.82,
    'P243': 0.3,
    'P244': 0,
    'P248': 0.82,
    'P249': 0.3,
    'P250': 0,
    'P251': 0.82,
    'P253': 0,
}

ZERODOSE = ["P278", "P282", "P286", "P244", "P250", "P253"]

TIME_DICT = {
    "P278": 4,
    "P279": 4,
    "P280": 4,
    "P282": 24,
    "P283": 24,
    "P284": 24,
    "P286": 48,
    "P287": 48,
    "P288": 48,
    "P242": 4,
    "P243": 4,
    "P244": 4,
    "P248": 24,
    "P249": 24,
    "P250": 24,
    "P251": 48,
    "P253": 48,
}

RAD_LABEL = {f"{DATA_BASE_PATH}Fe": 0, f"{DATA_BASE_PATH}X-RAY": 1}

THRESHOLD_TRAIN_SIZE = 0.7

SVM_TEST_SIZE = 0.3

BATCH_SIZE = 64

ADAM_B1 = 0.5
ADAM_B2 = 0.999
LR = 0.0001
EPOCHS = 150
NUM_CLASS = 2

CNN_PATH = 'checkpoint.pt1'
CNN_LOG_FILE = 'cnn_log.log'
CNN_LOSS_PLOT = 'cnn_loss.png'
CNN_ACCU_PLOT = 'cnn_accu.png'
CNN_CONFUSION = 'confusion_matrix_time.png'
