import logging
from pathlib import Path
# --- 
# ollama_model_index = "llama3.2"
ollama_model_index = "llama3.2-vision"
yolo_model_index = "yolov8n"
# yolo_model_index = "yolo11n"
train_data = "data.yaml"
whisper_model_index = "base"
sound_input = "input.wav"
rate = 44100
duration = 5
camera_index = 1
verbose = False
confidence_threshold = 0.5
chunk_size = 1024
threshold = 50
silence = 2.0


# --- Path Settings ---
BASE_DIR = Path( __file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"



# --- Ensure directories exist ---
DATA_DIR.mkdir(parents = True, exist_ok = True)
(DATA_DIR / "raw").mkdir(parents = True, exist_ok = True)
(DATA_DIR / "result").mkdir(parents = True, exist_ok = True)
LOGS_DIR.mkdir(parents = True, exist_ok = True)
(DATA_DIR / "yaml").mkdir(parents = True, exist_ok = True)
(DATA_DIR / "weights").mkdir(parents = True, exist_ok = True)
(DATA_DIR / "user_sound").mkdir(parents = True, exist_ok = True)

# --- Model Setting ---
YOLO_MODEL_NAME = yolo_model_index
MODEL_PATH = MODELS_DIR / YOLO_MODEL_NAME
OLLAMA_MODEL_NAME = ollama_model_index
WHISPER_MODEL_NAME = whisper_model_index

# ---BOX Settings ---
CONFIDENCE_THRESHOLD = confidence_threshold

# --- Camera Settings ---
CAMERA_INDEX = camera_index
VERBOSE_STATUS = verbose

# --- VAD setting ---
THRESHOLD = threshold
SILENCE_LIMIT = silence
CHUNK_SIZE = chunk_size
ASK_HISTORY = []

# --- Sound setting ---
SAMPLE_RATE = rate
DURATION = duration
WAV_OUTPUT_PATH = DATA_DIR / "user_sound" / sound_input

# --- Logging Settings ---
LOG_FILE = LOGS_DIR / "app.log"
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOG_LEVEL = logging.INFO

# --- train Setup ---
YAML_PATH = DATA_DIR / "yaml"
WEIGHTS_PATH = DATA_DIR / "weights" 
INPUT_YAML_SMT = YAML_PATH / "SMT_v3i_yolov8" / train_data


