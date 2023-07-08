import matplotlib
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import os
import skimage
import IPython.display
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import clip
from collections import OrderedDict
import torch
from torchvision.datasets import CIFAR100

model, preprocess = clip.load("ViT-B/32")
model.cuda().eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

Compose([
    Resize(size=224, max_size=None, antialias=None),
    CenterCrop(size=224),
    ToTensor(),
    Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))])

# images in skimage to use and their textual descriptions

descriptions = {}
original_images = []
images = []
texts = []
data_dir = 'C:/keywords'
txt_path = 'C:/keywords/newDescriptions.txt'
labels_path = 'C:/filter/newLabels.txt'
max_token_size = 77
imshow_num = 4

with open(txt_path, 'r', encoding='utf-8') as f:
    content = f.read()
f.close()
i = 0
while i < len(content):
    if content[i:i + 5] == "<key>":
        i += 6
        fig = ""
        while content[i] != '<':
            fig += content[i]
            i += 1
        i += 15
        des = ""
        while content[i] != '<':
            des += content[i]
            i += 1
        fig = fig[:-1]
        des = des[:-1]
        descriptions[fig] = des
    else:
        i += 1
print(f'{descriptions}')
for filename in [filename for filename in os.listdir(data_dir) if
                 filename.endswith(".png") or filename.endswith(".jpg")]:
    name = os.path.splitext(filename)[0]
    if name not in descriptions:
        continue

    image = Image.open(os.path.join(data_dir, filename)).convert("RGB")
    original_images.append(image)
    images.append(preprocess(image))
    texts.append(descriptions[name])

image_input = torch.tensor(np.stack(images)).cuda()
text_tokens = clip.tokenize([desc for desc in texts]).cuda()

with torch.no_grad():
    image_features = model.encode_image(image_input).float()
    text_features = model.encode_text(text_tokens).float()

image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T

count = imshow_num

# print(f'{len(original_images)}')
plt.figure(figsize=(10, 7))
plt.imshow(similarity, vmin=0, vmax=1)
# plt.colorbar()
plt.yticks(range(count), texts[0:imshow_num], fontsize=9)
plt.xticks([])
for i, image in enumerate(original_images):
    if i > imshow_num-1:
        break
    plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
for x in range(similarity.shape[1]):
    if x>imshow_num-1:
        continue
    for y in range(similarity.shape[0]):
        if y>imshow_num-1:
            continue
        plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=6)

for side in ["left", "top", "right", "bottom"]:
    plt.gca().spines[side].set_visible(False)

plt.xlim([-0.5, count - 0.5])
plt.ylim([count + 0.5, -2])

plt.title("Cosine similarity between text and image features", size=10)

labels = []
with open(labels_path, 'r', encoding='utf-8') as f1:
    line = f1.readline()
    while line:
        if len(line) < max_token_size:
            labels.append(line)
        line = f1.readline()
f1.close()
# print(f'{labels}')
text_descriptions = [f'{label}' for label in labels]
text_tokens = clip.tokenize(text_descriptions).cuda()

with torch.no_grad():
    text_features = model.encode_text(text_tokens).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)

text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
top_probs, top_labels = text_probs.cpu().topk(5, dim=-1)

plt.figure(figsize=(16, 16))

for i, image in enumerate(original_images):
    if i > 3:
        break
    plt.subplot(4, 4, 2 * i + 1)
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(4, 4, 2 * i + 2)
    y = np.arange(top_probs.shape[-1])
    plt.grid()
    plt.barh(y, top_probs[i])
    plt.gca().invert_yaxis()
    plt.gca().set_axisbelow(True)
    plt.yticks(y, [labels[index] for index in top_labels[i].numpy()])
    plt.xlabel("probability")

plt.subplots_adjust(wspace=0.5)
plt.show()
