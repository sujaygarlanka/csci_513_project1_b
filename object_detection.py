import cv2
import torch
from PIL import Image

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Images
# for f in 'zidane.jpg', 'bus.jpg':
#     torch.hub.download_url_to_file('https://ultralytics.com/images/' + f, f)  # download 2 images
# im1 = Image.open('zidane.jpg')  # PIL image
im1 = cv2.imread('./dist_4.3_ego_ado.png')[..., ::-1]  # OpenCV image (BGR to RGB)
im2 = cv2.imread('./dist_6.3_ego_ado.png')[..., ::-1]  # OpenCV image (BGR to RGB)
im3 = cv2.imread('./dist_25.8_ego_ado.png')[..., ::-1]  # OpenCV image (BGR to RGB)

# Inference
results = model([im1, im2, im3], size=640) # batch of images

# Results
results.print()  
results.show()  # or .show()

print(results.xyxy[1])  # im1 predictions (tensor)
# results.pandas().xyxy[0]  # im1 predictions (pandas)