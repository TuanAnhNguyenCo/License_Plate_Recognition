from huggingface_hub import PyTorchModelHubMixin
import requests
import albumentations as A
from torch import nn
from PIL import Image, ImageDraw, ImageFont
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import cv2
import torch
from paddleocr import PaddleOCR

class DETRModel(nn.Module,PyTorchModelHubMixin):
    def __init__(self,num_classes = 2,num_queries = 20):
        super(DETRModel,self).__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        
        self.model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=False)
        self.in_features = self.model.class_embed.in_features
        
        self.model.class_embed = nn.Linear(in_features=self.in_features,out_features=self.num_classes)
        self.model.num_queries = self.num_queries
        
    def forward(self,images):
        return self.model(images)


class LicensePlateInference():
    def __init__(self,device = 'cpu'):
        self.transforms = A.Compose([A.Resize(height=512, width=512, p=1.0),
                    ToTensorV2(p=1.0)], 
                    )
        self.detection_model = DETRModel.from_pretrained("NCTuanAnh/vietnamese_license_plate_detection_DeRT").to(device)
        self.detection_model.eval()
        self.ocr_model = PaddleOCR(lang='en',cls = True)
        self.device = device
        
    def preprocess(self,img):
        if isinstance(img,str): # input is string
            image = cv2.imread(img)
        else:
            image = img # input is np array
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255
        if self.transforms:
            sample = {
                'image': image,
            }
            sample = self.transforms(**sample)
            image = sample['image'].unsqueeze(dim = 0)
        
        return image.to(self.device)
    
    def inference(self,image,output_url = 'output.png'):
        frame = self.preprocess(image)
        with torch.no_grad():
            output = self.detection_model(frame) # infer
        out = (output['pred_logits'].argmax(dim = -1) == 0).bool()
        bboxes = output['pred_boxes'][out].detach().cpu().numpy() # get bboxes
        
        image = cv2.imread(image) if isinstance(image,str) else image
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)


        height,width,c = image.shape
        for x,y,w,h in bboxes:
            x1 = x - w/2
            y1 = y - h/2
            x2 = x + w/2
            y2 = y + h/2
            x1,x2 = int(x1*width),int(x2*width)
            y1,y2 = int(y1*height),int(y2*height)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # recognition
            img = image[y1:y2,x1:x2]
            result = self.ocr_model.ocr(img)
            y = y1 - 30 if len(result[0]) > 1 else y1 - 10
            for idx,out in enumerate(result[0]):
                text = out[1][0]
                cv2.putText(image, text, (x1, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                y +=20
                
        cv2.imwrite(output_url, image)
        return image
    
    def video_inference(self,input_url,output_url = 'demo.mp4'):
        # Load video
        video_input = cv2.VideoCapture(input_url)
      

        # Check if video is successfully loaded
        if not video_input.isOpened():
            print("Error: Could not open video file.")
            exit()

        # Get video properties
        frame_width = int(video_input.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_input.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video_input.get(cv2.CAP_PROP_FPS))
        frame_count = int(video_input.get(cv2.CAP_PROP_FRAME_COUNT))

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
        output_video = cv2.VideoWriter(output_url, fourcc, fps, (frame_width, frame_height))
        while True:
            ret_input, frame_input = video_input.read()
            if not ret_input:
                break
            frame = self.preprocess(frame_input)
            with torch.no_grad():
                output = self.detection_model(frame) # infer
            out = (output['pred_logits'].argmax(dim = -1) == 0).bool() # class 0 <==> license plate
            bboxes = output['pred_boxes'][out].detach().cpu().numpy()  
            
            image = cv2.cvtColor(frame_input,cv2.COLOR_BGR2RGB)

            height,width,c = image.shape
            for x,y,w,h in bboxes:
                x1 = x - w/2
                y1 = y - h/2
                x2 = x + w/2
                y2 = y + h/2
                x1,x2 = int(x1*width),int(x2*width)
                y1,y2 = int(y1*height),int(y2*height)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # recognition
                img = image[y1:y2,x1:x2]
                result = self.ocr_model.ocr(img)
                try:
                    y = y1 - 30 if len(result[0]) > 1 else y1 - 10
                    for idx,out in enumerate(result[0]):
                        text = out[1][0]
                        cv2.putText(image, text, (x1, y),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                        y +=20
                except:
                    pass
            
            image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
                # Write the frame into the output video file
            output_video.write(image)
        
        # Release video objects
        video_input.release()
        output_video.release()

        