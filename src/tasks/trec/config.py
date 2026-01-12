import os

# Models
STUDENT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
TEACHER_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# Model Generation Parameters
STUDENT_MODEL_KWARGS = {
    "max_new_tokens": 128,  
    "temperature": 0.3,     
    "top_p": 0.9,
}

TEACHER_MODEL_KWARGS = {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
}

# Training Settings
TRAIN_BATCH_SIZE = 8  
VAL_BATCH_SIZE = 8
MAX_STEPS = 12 

# Data Sizes
TRAIN_SIZE = 100 
VAL_SIZE = 50 
TEST_SIZE = 100

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(os.getcwd(), "outputs", "trec", "ckpt")
OUTPUT_DIR = os.path.join(os.getcwd(), "outputs", "trec")