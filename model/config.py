# Model config ======
RUN_NAME        = 'unetv1'
N_CLASSES       = 1
INPUT_SIZE      = 128
EPOCHS          = 10
LEARNING_RATE   = 0.0001
START_FRAME     = 16
DROP_RATE       = 0.5
REDUCE_RATE     = 0.8

# Data config =======
SAVE_PATH       = './model/'
DATA_PATH       = './data/'
UPLOAD_FOLDER   = './uploads/'
SUPPORT_FORMAT  = ['mp3', 'mp4', 'wav']

NOISE_DOMAINS   = ['vacuum_cleaner', 'clapping', 'fireworks', 'door_wood_knock', 'engine', 'mouse_click', 
                    'clock_alarm', 'wind', 'keyboard_typing', 'footsteps', 'car_horn', 'drinking_sipping', 'snoring', 
                    'breathing', 'toilet_flush', 'clock_tick', 'washing_machine', 'rain', 'rooster', 'laughing']

RANDOM_SEED     = 42
VALID_RATIO     = 0.2
BATCH_SIZE      = 16
NUM_WORKERS     = 0

# Speech config =====
SAMPLE_RATE         = 8000
N_FFT               = 255
HOP_LENGTH_FFT      = 63
HOP_LENGTH_FRAME    = 8064
FRAME_LENGTH        = 8064
MIN_DURATION        = 1.0