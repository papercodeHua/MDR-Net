import os
import numpy as np
from PIL import Image


def compute(img_dir="DRIVE/aug/images"):
    img_channels = 3
    img_name_list = [i for i in os.listdir(img_dir)]
    cumulative_mean = np.zeros(img_channels)
    cumulative_std = np.zeros(img_channels)
    for img_name in img_name_list:
        img_path = os.path.join(img_dir, img_name)
        img = np.array(Image.open(img_path)) / 255.
        cumulative_mean += img.mean(axis=(0, 1))
        cumulative_std += img.std(axis=(0, 1))

    mean = cumulative_mean / len(img_name_list)
    std = cumulative_std / len(img_name_list)
    return mean, std


if __name__ == "__main__":
    mean, std = compute("DRIVE/aug/images")
    print(mean, std)
    mean1, std1 = compute("DRIVE/test/images")
    print(mean1, std1)
