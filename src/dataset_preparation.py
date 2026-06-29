import os
import yaml
import shutil
import random
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
from src.utils import logger

CLASSES = ["Person", "Helmet", "Safety Vest", "Gloves", "Face Mask", "Fire", "Smoke"]

def validate_image(image_path):
    """Verify if image file is valid and readable"""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False

def clean_and_validate_yolo_dataset(image_dir, label_dir):
    """
    Remove corrupted images and mismatching labels.
    Ensure label format is correct (class x_center y_center width height).
    """
    if not os.path.exists(image_dir) or not os.path.exists(label_dir):
        logger.warning(f"Paths do not exist: {image_dir} or {label_dir}")
        return
        
    images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    removed_count = 0
    
    for img_name in images:
        base_name = os.path.splitext(img_name)[0]
        img_path = os.path.join(image_dir, img_name)
        lbl_path = os.path.join(label_dir, base_name + ".txt")
        
        # Check image validity
        if not validate_image(img_path):
            logger.info(f"Invalid image found, deleting: {img_path}")
            os.remove(img_path)
            if os.path.exists(lbl_path):
                os.remove(lbl_path)
            removed_count += 1
            continue
            
        # Check matching label
        if not os.path.exists(lbl_path):
            # If label does not exist, create empty label (negative background sample)
            with open(lbl_path, 'w') as f:
                pass
            continue
            
        # Validate label contents
        valid_lines = []
        try:
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_idx = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                    # Check values are normalized (0 to 1) and class index is valid
                    if 0 <= cls_idx < len(CLASSES) and all(0 <= c <= 1 for c in coords):
                        valid_lines.append(line)
            
            # Rewrite clean labels
            with open(lbl_path, 'w') as f:
                f.writelines(valid_lines)
        except Exception as e:
            logger.error(f"Error checking label {lbl_path}: {e}")
            
    logger.info(f"Cleaned dataset in {image_dir}. Removed {removed_count} corrupted images.")

def convert_voc_to_yolo(xml_dir, img_dir, output_label_dir, class_mapping):
    """
    Convert Pascal VOC annotations (XML) to YOLO format (TXT).
    class_mapping: dict mapping VOC class name to YOLO index
    """
    os.makedirs(output_label_dir, exist_ok=True)
    
    for xml_file in os.listdir(xml_dir):
        if not xml_file.endswith('.xml'):
            continue
            
        xml_path = os.path.join(xml_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        
        filename = root.find('filename').text
        txt_name = os.path.splitext(xml_file)[0] + ".txt"
        txt_path = os.path.join(output_label_dir, txt_name)
        
        with open(txt_path, 'w') as out_file:
            for obj in root.iter('object'):
                cls_name = obj.find('name').text
                if cls_name not in class_mapping:
                    continue
                cls_id = class_mapping[cls_name]
                
                xmlbox = obj.find('bndbox')
                xmin = float(xmlbox.find('xmin').text)
                xmax = float(xmlbox.find('xmax').text)
                ymin = float(xmlbox.find('ymin').text)
                ymax = float(xmlbox.find('ymax').text)
                
                # YOLO format: x_center, y_center, width, height normalized
                x_center = (xmin + xmax) / 2.0 / w
                y_center = (ymin + ymax) / 2.0 / h
                width = (xmax - xmin) / w
                height = (ymax - ymin) / h
                
                out_file.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def merge_datasets(src_dirs, dest_dir):
    """
    Merge multiple datasets into a single target directory.
    src_dirs: list of paths to dataset roots (each containing train/valid/test directories)
    dest_dir: root directory where datasets are combined
    """
    splits = ['train', 'valid', 'test']
    subdirs = ['images', 'labels']
    
    for split in splits:
        for subdir in subdirs:
            os.makedirs(os.path.join(dest_dir, split, subdir), exist_ok=True)
            
    for src in src_dirs:
        logger.info(f"Merging dataset {src} into {dest_dir}...")
        for split in splits:
            src_img_dir = os.path.join(src, split, 'images')
            src_lbl_dir = os.path.join(src, split, 'labels')
            
            if not os.path.exists(src_img_dir):
                continue
                
            for file_name in os.listdir(src_img_dir):
                src_img_path = os.path.join(src_img_dir, file_name)
                dest_img_path = os.path.join(dest_dir, split, 'images', file_name)
                shutil.copy2(src_img_path, dest_img_path)
                
                # Match label
                base_name = os.path.splitext(file_name)[0]
                src_lbl_path = os.path.join(src_lbl_dir, base_name + ".txt")
                dest_lbl_path = os.path.join(dest_dir, split, 'labels', base_name + ".txt")
                if os.path.exists(src_lbl_path):
                    shutil.copy2(src_lbl_path, dest_lbl_path)
                else:
                    # Write empty label if missing
                    with open(dest_lbl_path, 'w') as f:
                        pass
                        
    logger.info("Dataset merge operation completed.")

def write_data_yaml(dataset_dir, output_path):
    """Create data.yaml for YOLO training"""
    abs_dataset_dir = os.path.abspath(dataset_dir)
    data = {
        'path': abs_dataset_dir,
        'train': os.path.join('train', 'images'),
        'val': os.path.join('valid', 'images'),
        'test': os.path.join('test', 'images'),
        'names': {i: name for i, name in enumerate(CLASSES)}
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    logger.info(f"Generated YAML configuration at {output_path}")

def generate_synthetic_dataset(base_dir):
    """
    Generate a simple colored mockup image dataset with label text files.
    This creates simulated annotated data to let the training pipeline execute successfully.
    """
    logger.info(f"Generating synthetic dataset inside: {base_dir}")
    splits = {
        'train': 15, # 15 images
        'valid': 5,  # 5 images
        'test': 3    # 3 images
    }
    
    for split, count in splits.items():
        img_dir = os.path.join(base_dir, split, 'images')
        lbl_dir = os.path.join(base_dir, split, 'labels')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        
        for i in range(count):
            img_name = f"synthetic_{split}_{i}.jpg"
            lbl_name = f"synthetic_{split}_{i}.txt"
            
            # Create a 256x256 image with solid colored background
            img = Image.new('RGB', (256, 256), color=(
                random.randint(50, 150),
                random.randint(50, 150),
                random.randint(50, 150)
            ))
            draw = ImageDraw.Draw(img)
            
            # Place 1 to 3 items in the image
            annotations = []
            num_objects = random.randint(1, 3)
            for _ in range(num_objects):
                class_id = random.randint(0, len(CLASSES) - 1)
                
                # Pick box coords in 256x256 scale
                w_box = random.randint(40, 80)
                h_box = random.randint(40, 100)
                x_center_raw = random.randint(w_box // 2 + 10, 240 - w_box // 2)
                y_center_raw = random.randint(h_box // 2 + 10, 240 - h_box // 2)
                
                # Calculate corners
                x1 = x_center_raw - w_box // 2
                y1 = y_center_raw - h_box // 2
                x2 = x_center_raw + w_box // 2
                y2 = y_center_raw + h_box // 2
                
                # Draw on synthetic image to make it look like a detection subject
                # e.g., Helmet draws orange rectangle, vest draws neon green, person draws blue
                color_map = {
                    0: (0, 0, 255),    # Person - Blue
                    1: (255, 128, 0),  # Helmet - Orange
                    2: (128, 255, 0),  # Vest - Neon Green
                    3: (255, 255, 0),  # Gloves - Yellow
                    4: (0, 255, 255),  # Mask - Cyan
                    5: (255, 0, 0),    # Fire - Red
                    6: (192, 192, 192) # Smoke - Grey
                }
                draw.rectangle([x1, y1, x2, y2], outline=color_map.get(class_id, (255, 255, 255)), width=2)
                
                # Normalized YOLO values
                x_center = x_center_raw / 256.0
                y_center = y_center_raw / 256.0
                width = w_box / 256.0
                height = h_box / 256.0
                annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                
            img.save(os.path.join(img_dir, img_name))
            
            with open(os.path.join(lbl_dir, lbl_name), 'w') as f:
                f.write("\n".join(annotations))
                
    # Create data.yaml
    write_data_yaml(base_dir, os.path.join(base_dir, 'data.yaml'))
    logger.info("Synthetic dataset creation finished.")

if __name__ == "__main__":
    # If run directly, generate synthetic dataset for test purposes
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(workspace_dir, 'dataset', 'combined_dataset')
    generate_synthetic_dataset(target_dir)
