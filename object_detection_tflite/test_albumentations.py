import albumentations as A
import numpy as np
from PIL import Image
# Declare an augmentation pipeline
transform = A.Compose([
    A.Blur(p=1.0)
])

# Read an image with OpenCV and convert it to the RGB colorspace

image = np.array(Image.open("data/IP-0000001.png"))
# Augment an image
transformed = transform(image=image)
transformed_image = transformed["image"]


out = np.concatenate([image, transformed_image], axis=1)

image = Image.fromarray(out)

image.show()



