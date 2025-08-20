# Real-Time River Segmentation Model
It contains a fine-tuned deep learning model for pixel-perfect segmentation of river water from aerial imagery. Its primary goal is to detect the "river crest" (the boundary between water and land) and output its precise coordinates.

## What's Inside
mobilenet_unet_final.pt: The final, fine-tuned PyTorch model file. This is the "brain" of the operation.

test_model.py: A Python script to run the model on a folder of test images.

test_images/: A folder where you should place your input images (.jpg, .png).

test_outputs/: The folder where the script will save all results.

## Setup
To run this project on your local machine, you need to install the required Python libraries.

Clone the repository.

Install dependencies: Open your terminal in the project folder and run:

``` Bash

pip install torch torchvision opencv-python numpy albumentations
```
## How to Use
Add Test Images: Place the images you want to analyze into the test_images/ folder.

Run the Script: Open your terminal in the project folder and execute the script:

```Bash

python test_model.py
```
The script will process every image in the test_images folder and generate two output files for each one inside the test_outputs/ directory:

An annotated image (e.g., test_image_1.jpg) showing the detected water as a blue overlay and the crest edge as a green line.

A JSON file (e.g., test_image_1.json) containing the timestamp, detection confidence, and a precise list of pixel coordinates for the river's edge. This data is ready for analysis.