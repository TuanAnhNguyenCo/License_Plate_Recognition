# Files
- detection.py: This file consists of inference code for this project
- vietnamese-license-plate-detection-with-dert.ipynb: This file contains the training code of License Plate Detection with DeRT

# Lib:
- Install If you don't have:
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
- In this project, I used the DeRT model to detect the license plate and then used paddle OCR to recognize the license plate. But you can train vehicle detection and then use paddle OCR.

# Data:
- You can download here: https://github.com/trungdinh22/License-Plate-Recognition

# Training:
- You can use vietnamese-license-plate-detection-with-dert.ipynb file to train the license plate detection or something. Note: I set up queries = 20 and n_classes = 2. You can customize it in your way.

# Inference:
 ```
  pipeline = LicensePlateInference(device = 'cpu') # initialize pipeline
  # If you want to infer with image URL or np array you can use the following function
  image = pipeline.inference('input.jpeg',output_url='output.jpeg')
  # or
  image = pipeline.inference(frame,output_url='output.jpeg')
  # If you want to infer with video you can use the following function
  pipeline.video_inference('input.mp4','output.mp4')
 ```
- Please take note of the image generated by the pipeline.inference includes bounding boxes and text 
# Example:
![Example](https://github.com/TuanAnhNguyenCo/License_Plate_Recognition/blob/main/output1.jpeg)
![Example](https://github.com/TuanAnhNguyenCo/License_Plate_Recognition/blob/main/output.jpeg)
 
