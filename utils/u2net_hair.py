
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50

model = deeplabv3_resnet50(pretrained=True).eval()

def segment_hair(image):
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((320, 320)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    mask = output.argmax(0).byte().cpu().numpy()
    hair_mask = (mask == 15).astype(np.uint8)
    hair_mask = cv2.resize(hair_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    return hair_mask
