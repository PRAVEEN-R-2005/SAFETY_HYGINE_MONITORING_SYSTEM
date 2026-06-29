import os
import shutil
import torch
from ultralytics import YOLO
from src.utils import logger
from config import Config

def train_yolo_model(data_yaml_path, epochs=5, batch_size=8, imgsz=256):
    """
    Train a YOLOv11 model on the dataset specified in data.yaml.
    For demonstration/testing, epochs are set small. In production, set epochs to 50-100.
    """
    logger.info("Initializing YOLOv11 Model Training...")
    
    # Ensure models directory exists
    models_dir = os.path.dirname(Config.MODEL_PATH)
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Load pretrained YOLOv11 model (yolo11n.pt nano model)
    # The 'yolo11n.pt' weight file will be downloaded automatically by Ultralytics if not locally cached.
    try:
        model = YOLO('yolo11n.pt')
        logger.info("Loaded pretrained yolo11n.pt weight successfully.")
    except Exception as e:
        logger.error(f"Error loading yolo11n.pt: {e}. Falling back to initializing a new model.")
        model = YOLO() # fallback
        
    # 2. Check for GPU acceleration
    device = 0 if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device for training: {device} (CUDA available: {torch.cuda.is_available()})")
    
    # 3. Start training
    logger.info(f"Starting training on {data_yaml_path} for {epochs} epochs (imgsz={imgsz}, batch={batch_size})...")
    try:
        results = model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            project='runs/detect',
            name='safety_training_run',
            exist_ok=True
        )
        logger.info("YOLOv11 model training process completed.")
        
        # 4. Save best and last weights to models/
        run_weights_dir = os.path.join('runs', 'detect', 'safety_training_run', 'weights')
        best_run_weight = os.path.join(run_weights_dir, 'best.pt')
        last_run_weight = os.path.join(run_weights_dir, 'last.pt')
        
        if os.path.exists(best_run_weight):
            shutil.copy2(best_run_weight, Config.MODEL_PATH)
            shutil.copy2(best_run_weight, os.path.join(models_dir, 'best.pt'))
            logger.info(f"Copied best model weights to: {Config.MODEL_PATH}")
        else:
            logger.warning("best.pt was not found in training output directory.")
            
        if os.path.exists(last_run_weight):
            shutil.copy2(last_run_weight, os.path.join(models_dir, 'last.pt'))
            logger.info(f"Copied last model weights to: {os.path.join(models_dir, 'last.pt')}")
            
        # 5. Copy metrics visual files to models directory for UI reference
        training_project_dir = os.path.join('runs', 'detect', 'safety_training_run')
        metric_files = ['results.png', 'confusion_matrix.png', 'F1_curve.png', 'P_curve.png', 'R_curve.png']
        for file in metric_files:
            src_file = os.path.join(training_project_dir, file)
            if os.path.exists(src_file):
                shutil.copy2(src_file, os.path.join(models_dir, file))
                logger.info(f"Copied metric plot {file} to models/")
                
        return True
    except Exception as e:
        logger.error(f"Training pipeline execution failed: {e}")
        return False

if __name__ == "__main__":
    # Locate data.yaml inside combined_dataset
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_yaml = os.path.join(workspace_dir, 'dataset', 'combined_dataset', 'data.yaml')
    
    # Run training for 1 epoch to quickly verify training works
    if not os.path.exists(data_yaml):
        logger.warning("Combined dataset YAML not found. Running synthetic dataset preparation first.")
        from src.dataset_preparation import generate_synthetic_dataset
        synthetic_dir = os.path.join(workspace_dir, 'dataset', 'combined_dataset')
        generate_synthetic_dataset(synthetic_dir)
        
    train_yolo_model(data_yaml, epochs=1, batch_size=4, imgsz=128)
