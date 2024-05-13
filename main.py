from detection import LicensePlateInference



pipeline  = LicensePlateInference(device='cpu')
pipeline.inference('input1.jpeg',output_url='output1.jpeg')
# pipeline.video_inference('input.mp4','output.mp4')

