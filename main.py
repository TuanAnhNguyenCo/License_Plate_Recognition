from detection import LicensePlateInference



pipeline  = LicensePlateInference(device='cuda:1')
pipeline.inference('input.jpeg',output_url='output.jpeg')
pipeline.video_inference('input.mp4','output.mp4')

