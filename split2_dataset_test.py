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
import get_des_from_txt as gt


model, preprocess = clip.load("ViT-B/32")

print(clip.tokenize("thyroid cytology. Ciliated respiratory epithelial cells. These may be obtained from inadvertent sampling of the trachea during a thyroid FNA. (ThinPrep, Papanicolaou.) "))


# images in skimage to use and their textual descriptions
descriptions = {
    "page": "a page of text about segmentation",
    "chelsea": "a facial photo of a tabby cat",
    "astronaut": "a portrait of an astronaut with the American flag",
    "rocket": "a rocket standing on a launchpad",
    "motorcycle_right": "a red motorcycle standing in a garage",
    "camera": "a person looking at a camera on a tripod",
    "horse": "a black-and-white silhouette of a horse",
    "coffee": "a cup of coffee on a saucer"
}

original_images = []
images=[]
texts=[]


image_list=gt.get_img()
text_dict=gt.get_des()

#print(text_dict.keys())
for img in image_list:
    image=Image.open(img).convert("RGB")
    img_name=img.split('\\')[-1]
    original_images.append(image)
    images.append(preprocess(image))
    texts.append(text_dict[img_name])

'''for filename in [filename for filename in os.listdir(skimage.data_dir) if
                 filename.endswith(".png") or filename.endswith(".jpg")]:
    name = os.path.splitext(filename)[0]
    if name not in descriptions:
        continue

    image = Image.open(os.path.join(skimage.data_dir, filename)).convert("RGB")

    plt.subplot(2, 4, len(images) + 1)
    plt.imshow(image)
    plt.title(f"{filename}\n{descriptions[name]}")
    plt.xticks([])
    plt.yticks([])

    original_images.append(image)
    images.append(preprocess(image))
    texts.append(descriptions[name])'''

image_input = torch.tensor(np.stack(images)).cuda()
text_tokens = clip.tokenize(["This is " + desc for desc in texts]).cuda()
print(text_tokens)

with torch.no_grad():
    image_features = model.encode_image(image_input).float()
    text_features = model.encode_text(text_tokens).float()

image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
print(len(image_features),len(image_features[0]))
print(len(text_features),len(text_features[0]))
similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T

count = len(descriptions)
print(similarity)


text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
print(text_probs)
top_probs, top_labels = text_probs.cpu().topk(5, dim=-1)
print(top_probs,top_labels)
