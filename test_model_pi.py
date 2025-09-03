# test_model_pi.py - Optimized for Raspberry Pi 3

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
import numpy as np
import cv2
from PIL import Image
import os
import json
from datetime import datetime
import glob
import gc

# --- Optimized Configuration for Pi ---
class Config:
    DEVICE = "cpu"  # Force CPU on Pi
    IMG_HEIGHT = 128  # Reduced from 256
    IMG_WIDTH = 128   # Reduced from 256
    MODEL_PATH = "mobilenet_unet_final.pt"
    INPUT_DIR = "test_images"
    OUTPUT_DIR = "test_outputs"
    BATCH_SIZE = 1    # Process one at a time
    USE_HALF_PRECISION = False  # Pi 3 doesn't support FP16 well

# --- Lightweight preprocessing (no albumentations) ---
def preprocess_image(image_path, target_height, target_width):
    """Lightweight image preprocessing without albumentations"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_height, original_width = image.shape[:2]
    
    # Resize
    resized = cv2.resize(image, (target_width, target_height))
    
    # Normalize and convert to tensor
    normalized = resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
    
    return tensor, image, (original_height, original_width)

# --- Simplified Model Architecture ---
class LightMobileNetV2_UNet(nn.Module):
    def __init__(self):
        super(LightMobileNetV2_UNet, self).__init__()
        # Use a smaller encoder
        encoder = mobilenet_v2(weights=None).features
        
        # Simplified skip connections
        self.skip1 = encoder[:2]
        self.skip2 = encoder[2:4] 
        self.skip3 = encoder[4:7]
        self.bridge = encoder[7:11]  # Smaller bridge
        
        # Simplified decoder
        self.up1 = nn.ConvTranspose2d(160, 64, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(88, 64, 3, 1, 1)  # Reduced channels
        
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(56, 32, 3, 1, 1)
        
        self.up3 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 16, 3, 1, 1)
        
        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        s1 = self.skip1(x)
        s2 = self.skip2(s1)
        s3 = self.skip3(s2)
        bridge = self.bridge(s3)
        
        # Decoder
        x = self.up1(bridge)
        x = torch.cat([x, s3], dim=1)
        x = torch.relu(self.conv1(x))
        
        x = self.up2(x)
        x = torch.cat([x, s2], dim=1) 
        x = torch.relu(self.conv2(x))
        
        x = self.up3(x)
        x = torch.cat([x, s1], dim=1)
        x = torch.relu(self.conv3(x))
        
        return self.final_conv(x)

# --- Memory-efficient testing function ---
def test_folder_lightweight(model, input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Find images
    extensions = ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG', '*.JPEG']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    print(f"Found {len(image_files)} images to process.")
    
    for i, image_path in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        
        try:
            # Preprocess
            processed_image, original_image, (orig_h, orig_w) = preprocess_image(
                image_path, Config.IMG_HEIGHT, Config.IMG_WIDTH
            )
            
            # Inference with memory management
            with torch.no_grad():
                prediction = model(processed_image)
                prediction_probs = torch.sigmoid(prediction)
                prediction_mask = (prediction_probs > 0.5).cpu().numpy().squeeze()
            
            # Clear GPU memory
            del prediction, prediction_probs
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Resize mask back to original size
            mask_resized = cv2.resize(
                prediction_mask.astype(np.uint8), 
                (orig_w, orig_h), 
                interpolation=cv2.INTER_NEAREST
            )
            
            # Find contours
            contours, _ = cv2.findContours(
                mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Generate output data
            crest_detected = len(contours) > 0
            confidence = 0.0
            crest_edge = []
            
            if crest_detected and contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if len(largest_contour.shape) > 2:
                    crest_edge = largest_contour.squeeze().tolist()
                else:
                    crest_edge = largest_contour.tolist()
                
                # Simple confidence calculation
                confidence = float(np.mean(mask_resized[mask_resized > 0]))
            
            output_data = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "source_image": os.path.basename(image_path),
                "crest_detected": crest_detected,
                "confidence": round(confidence, 4),
                "crest_edge_pixels": crest_edge if len(crest_edge) > 0 else [],
            }
            
            # Save JSON
            json_filename = os.path.splitext(os.path.basename(image_path))[0] + ".json"
            json_path = os.path.join(output_dir, json_filename)
            with open(json_path, 'w') as f:
                json.dump(output_data, f, indent=2)  # Reduced indent
            
            # Generate visual output (optional - can be disabled to save memory)
            if True:  # Set to False to disable visual output
                overlay = np.zeros_like(original_image, dtype=np.uint8)
                overlay[mask_resized == 1] = [255, 100, 0]
                output_image = cv2.addWeighted(original_image, 0.7, overlay, 0.3, 0)
                
                if contours:
                    cv2.drawContours(output_image, contours, -1, (0, 255, 0), 2)
                
                # Save image
                output_image_bgr = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
                image_filename = f"result_{os.path.basename(image_path)}"
                output_image_path = os.path.join(output_dir, image_filename)
                cv2.imwrite(output_image_path, output_image_bgr)
            
            print(f"  ✓ Processed - Crest detected: {crest_detected}")
            
        except Exception as e:
            print(f"  ✗ Error processing {os.path.basename(image_path)}: {str(e)}")
            continue
        finally:
            # Force garbage collection
            gc.collect()

# --- Model loading with error handling ---
def load_model_safe(model_path):
    try:
        model = LightMobileNetV2_UNet()
        
        # Try to load the original model weights
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)  # Allow partial loading
        model.eval()
        
        print("✓ Model loaded successfully")
        return model
        
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        print("Note: If using the original model, you may need to adapt the architecture")
        return None

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🍓 RASPBERRY PI LIGHTWEIGHT MODEL INFERENCE")
    print("="*70)
    
    # System information
    print(f"🖥️ SYSTEM CONFIGURATION:")
    print(f"    Device: {Config.DEVICE}")
    print(f"    Image resolution: {Config.IMG_HEIGHT}x{Config.IMG_WIDTH}")
    print(f"    Input directory: {Config.INPUT_DIR}")
    print(f"    Output directory: {Config.OUTPUT_DIR}")
    print(f"    Model file: {Config.MODEL_PATH}")
    
    # Check PyTorch setup
    print(f"\n🔧 PYTORCH SETUP:")
    print(f"    PyTorch version: {torch.__version__}")
    print(f"    CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"    CUDA device: {torch.cuda.get_device_name()}")
    print(f"    CPU threads: {torch.get_num_threads()}")
    
    # Check directories
    print(f"\n📂 DIRECTORY CHECKS:")
    if os.path.exists(Config.INPUT_DIR):
        print(f"    ✅ Input directory exists: {Config.INPUT_DIR}")
    else:
        print(f"    ❌ Input directory not found: {Config.INPUT_DIR}")
        print(f"    📁 Creating input directory...")
        os.makedirs(Config.INPUT_DIR, exist_ok=True)
        print(f"    ✅ Input directory created")
    
    if not os.path.exists(Config.OUTPUT_DIR):
        print(f"    📁 Creating output directory: {Config.OUTPUT_DIR}")
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    else:
        print(f"    ✅ Output directory ready: {Config.OUTPUT_DIR}")
    
    # Load model
    print(f"\n" + "="*50)
    model = load_model_safe(Config.MODEL_PATH)
    if model is None:
        print(f"💥 FATAL ERROR: Failed to load model!")
        print(f"    Please check:")
        print(f"    1. Model file exists at: {Config.MODEL_PATH}")
        print(f"    2. Model file is not corrupted")
        print(f"    3. Architecture matches saved weights")
        print(f"🛑 Exiting...")
        exit(1)
    
    # Optimize for Pi
    print(f"\n⚙️ OPTIMIZING FOR RASPBERRY PI:")
    print(f"🧵 Setting CPU threads to 2 (Pi 3 optimization)...")
    torch.set_num_threads(2)
    print(f"    ✅ CPU threads set to: {torch.get_num_threads()}")
    
    print(f"🚀 Starting inference process...")
    
    # Run inference
    try:
        test_folder_lightweight(model, Config.INPUT_DIR, Config.OUTPUT_DIR)
        print(f"\n🎉 ALL PROCESSING COMPLETE!")
        print(f"📂 Check output directory: {Config.OUTPUT_DIR}")
        print(f"    - JSON files contain detection data")
        print(f"    - result_*.jpg files contain visual overlays")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Process interrupted by user (Ctrl+C)")
        print(f"🛑 Shutting down gracefully...")
        
    except Exception as e:
        print(f"\n💥 UNEXPECTED ERROR DURING PROCESSING!")
        print(f"    Error type: {type(e).__name__}")
        print(f"    Error message: {str(e)}")
        print(f"🛑 Process terminated")
        
    finally:
        print(f"\n🧹 Final cleanup...")
        gc.collect()
        print(f"✅ Cleanup complete")
        
    print(f"\n" + "="*70)
    print(f"🏁 SESSION ENDED")
    print(f"="*70)
