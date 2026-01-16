from ultralytics import YOLO
from src import config
import logging

def train_custom_model():
    # --- Log Loading ---
    logging.basicConfig(
        level = config.LOG_LEVEL,
        format = config.LOG_FORMAT,
        filename = config.LOG_FILE,
        filemode = 'a'
    )

    # --- Model Loading ---
    model = YOLO(str(config.MODEL_PATH))

    results = model.train(
        # data: path to your dataset .yaml file
        data = "config.INPUT_YAML_SMT",
        
        # epochs: How many times the model sees the entire dataset
        epochs = 50,
        
        # imgsz: Input image size (standard is 640) 
        imgsz = 640,

        # device: 'mps' leverages Apple Silicon GPU power
        device = "mps"
    )
    print