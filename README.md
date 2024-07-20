# Files
- detection.py:  This file contains the inference code for this project.
- vietnamese-license-plate-detection-with-dert.ipynb: This file contains the training code for License Plate Detection with DeRT.
# Lib:
- Install if you don't have them::
```!pip install lightning
   !pip install torchmetrics
   !pip install transformers
   !pip install -U albumentations
```
- If you have CUDA 9 or CUDA 10 installed on your machine, please run the following command to install
  ```
   python -m pip install paddlepaddle-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```
- If you have no available GPU on your machine, please run the following command to install the CPU version
  ```
   python -m pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```
- Install Paddle OCR Whl package:
  ```
     pip install "paddleocr>=2.0.1" # Recommend to use version 2.0.1+
  ```
# Introduce:
- In this project, I used the DeRT model to detect license plates and then PaddleOCR to recognize the text on them. You could also train a vehicle detection model instead of detecting license plates and still use PaddleOCR for text recognition. 

# Data:
- You can download here: https://github.com/trungdinh22/License-Plate-Recognition

# Training:
- Use the vietnamese-license-plate-detection-with-dert.ipynb file to train the license plate detection model. Note: I set queries = 20 and n_classes = 2. You can customize these values as needed.

# Inference:
 ```
  pipeline = LicensePlateInference(device = 'cpu') # Initialize the pipeline
  # Infer using an image URL or NumPy array:
  image = pipeline.inference('input.jpeg',output_url='output.jpeg')
  # or
  image = pipeline.inference(frame,output_url='output.jpeg')
  # Infer using a video:
  pipeline.video_inference('input.mp4','output.mp4')
 ```
# Example:
![Example](https://github.com/TuanAnhNguyenCo/License_Plate_Recognition/blob/main/output1.jpeg)
![Example](https://github.com/TuanAnhNguyenCo/License_Plate_Recognition/blob/main/output.jpeg)
 
