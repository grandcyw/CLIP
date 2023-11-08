# -- coding: utf-8 --
# @Time : 2023/11/8 9:37
# @Author : 王川远
# @Email : 3030764269@qq.com
# @File : wcy.py
# @Software: PyCharm
import torch
import clip
from PIL import Image
import time


t0=time.time()
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
t1=time.time()
print(t1-t0)
image = preprocess(Image.open("4.png")).unsqueeze(0).to(device)
t2=time.time()
print(t2-t1)
text = clip.tokenize(["a painting", "medical cells", "thyroid cytology","thyroid cytology. Ciliated respiratory epithelial cells. These may be obtained from inadvertent sampling of the trachea during a thyroid FNA. (ThinPrep, Papanicolaou.) ","thyroid cytology.Acute thyroiditis. Marked acute inflammation and debris are seen, but follicular cells and colloid are absent. (Smear, Papanicolaou.) "]).to(device)
t3=time.time()
print(t3-t2)
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    logits_per_image, logits_per_text = model(image, text)
    print(logits_per_image)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
