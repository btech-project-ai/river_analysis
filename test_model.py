# test_model.py

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
import numpy as np
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import json
from datetime import datetime
import glob

# --- Configuration ---
class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    IMG_HEIGHT = 256
    IMG_WIDTH = 256
    MODEL_PATH = "mobilenet_unet_final.pt"
    # Directories for batch processing
    INPUT_DIR = "test_images"
    OUTPUT_DIR = "test_outputs"

# --- Re-define the PyTorch Model Architecture (Same as before) ---
class MobileNetV2_UNet(nn.Module):
    def __init__(self):
        super(MobileNetV2_UNet, self).__init__()
        self.encoder = mobilenet_v2(weights=None).features
        
        self.skip1 = self.encoder[:2]
        self.skip2 = self.encoder[2:4]
        self.skip3 = self.encoder[4:7]
        self.skip4 = self.encoder[7:14]
        self.bridge = self.encoder[14:18]

        self.up1 = nn.ConvTranspose2d(320, 96, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(nn.Conv2d(192, 96, 3, 1, 1, bias=False), nn.ReLU(inplace=True))
        self.up2 = nn.ConvTranspose2d(96, 32, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1, bias=False), nn.ReLU(inplace=True))
        self.up3 = nn.ConvTranspose2d(32, 24, kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(nn.Conv2d(48, 24, 3, 1, 1, bias=False), nn.ReLU(inplace=True))
        self.up4 = nn.ConvTranspose2d(24, 16, kernel_size=2, stride=2)
        self.conv4 = nn.Sequential(nn.Conv2d(32, 16, 3, 1, 1, bias=False), nn.ReLU(inplace=True))
        self.final_up = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        s1, s2, s3, s4 = self.skip1(x), self.skip2(self.skip1(x)), self.skip3(self.skip2(self.skip1(x))), self.skip4(self.skip3(self.skip2(self.skip1(x))))
        bridge = self.bridge(s4)
        x = self.up1(bridge); x = torch.cat([x, s4], dim=1); x = self.conv1(x)
        x = self.up2(x); x = torch.cat([x, s3], dim=1); x = self.conv2(x)
        x = self.up3(x); x = torch.cat([x, s2], dim=1); x = self.conv3(x)
        x = self.up4(x); x = torch.cat([x, s1], dim=1); x = self.conv4(x)
        x = self.final_up(x)
        return self.final_conv(x)

# --- Main Testing Function ---
def test_folder(model, input_dir, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all images in the input directory
    image_files = glob.glob(os.path.join(input_dir, '*.jpg')) + \
                  glob.glob(os.path.join(input_dir, '*.png')) + \
                  glob.glob(os.path.join(input_dir, '*.jpeg'))

    print(f"Found {len(image_files)} images to process.")

    for image_path in image_files:
        print(f"\n--- Processing: {os.path.basename(image_path)} ---")
        
        # Load and preprocess image
        image = np.array(Image.open(image_path).convert("RGB"))
        original_height, original_width, _ = image.shape
        transform = A.Compose([
            A.Resize(height=Config.IMG_HEIGHT, width=Config.IMG_WIDTH),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
            ToTensorV2(),
        ])
        processed_image = transform(image=image)["image"].unsqueeze(0).to(Config.DEVICE)

        # Run Inference
        with torch.no_grad():
            prediction = model(processed_image)
            prediction_probs = torch.sigmoid(prediction)
            prediction_mask = (prediction_probs > 0.5).cpu().numpy().squeeze()

        # Resize mask and find contours
        mask_resized = cv2.resize(prediction_mask.astype(np.uint8), (original_width, original_height), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # --- Generate JSON Output ---
        crest_detected = len(contours) > 0
        crest_edge = []
        if crest_detected:
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            # Squeeze to remove unnecessary dimensions and convert to a simple list
            crest_edge = largest_contour.squeeze().tolist()

        # Convert probability tensor to numpy array before applying the mask
        probs_np = prediction_probs.cpu().numpy().squeeze()
        confidence = float(np.mean(probs_np[prediction_mask])) if crest_detected else 0.0 if crest_detected else 0.0
        
        output_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "source_image": os.path.basename(image_path),
            "crest_detected": crest_detected,
            "confidence": round(confidence, 4),
            "crest_edge_pixels": crest_edge,
        }
        
        # Save JSON file
        json_filename = os.path.splitext(os.path.basename(image_path))[0] + ".json"
        json_path = os.path.join(output_dir, json_filename)
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=4)
        print(f"Saved JSON output to: {json_path}")
        
        # --- Generate Visual Output ---
        overlay = np.zeros_like(image, dtype=np.uint8)
        overlay[mask_resized == 1] = (255, 100, 0)
        output_image = cv2.addWeighted(image, 1.0, overlay, 0.4, 0)
        if contours:
            cv2.drawContours(output_image, contours, -1, (0, 255, 0), 2)
        
        # Save annotated image
        output_image_bgr = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        image_filename = os.path.basename(image_path)
        image_path = os.path.join(output_dir, image_filename)
        cv2.imwrite(image_path, output_image_bgr)
        print(f"Saved annotated image to: {image_path}")

if __name__ == "__main__":
    print(f"--- Loading Model from {Config.MODEL_PATH} ---")
    model = MobileNetV2_UNet().to(Config.DEVICE)
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=torch.device(Config.DEVICE)))
    model.eval()
    
    test_folder(model, Config.INPUT_DIR, Config.OUTPUT_DIR)