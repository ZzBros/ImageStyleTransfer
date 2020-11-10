from torchvision import transforms
import cv2
import torch

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# devcice = torch.device('cpu')
def loadImg(imagePath, transform, shape=None):
    img = cv2.imread(imagePath, -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if shape:
        img = cv2.resize(img, shape, interpolation=cv2.INTER_AREA)
    
    img = transform(img).unsqueeze(0)
    return img.to(dev)