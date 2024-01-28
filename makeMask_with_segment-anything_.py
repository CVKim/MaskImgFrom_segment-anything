import torch
from PIL import Image
import requests

import numpy as np
import cv2
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

from PIL import Image
from transformers import SamModel, SamProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

raw_image.show()

input_points = [[[450, 600]]]  # 2D location of a window in the image

inputs = processor(raw_image, input_points=input_points, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

masks = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
)

for i, mask_info in enumerate(masks):
    # mask_data = mask_info['Segmentation']

    # 마스크 데이터를 uint8 타입으로 변환하고 255를 곱하여 바이너리 이미지를 생성합니다
    mask_image = (mask_info.astype(np.uint8) * 255)

    # 이미지 파일로 저장
    cv2.imwrite(os.path.join("tempss.bmp", f"mask_{i}.png"), mask_image)

    # 이미지 저장
    mask_image.save(f"mask_{i}.png")

scores = outputs.iou_scores
print("IOU Scores:", scores)
