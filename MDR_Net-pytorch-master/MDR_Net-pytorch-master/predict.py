import os
import time
from torch import nn
import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from model.MDR_Net import MDR_Net


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    classes = 1  # exclude background
    weights_path = "./save_weights/best_model.pth"
    img_path = "DRIVE/test/images/09_test.tif"
    roi_mask_path = "DRIVE/test/mask/09_test_mask.gif"
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."
    assert os.path.exists(roi_mask_path), f"image {roi_mask_path} not found."

    mean1 = (0.345, 0.345, 0.345)
    std1 = (0.252, 0.252, 0.252)

    # get devices
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = MDR_Net(in_channels=3, num_classes=classes, base_c=36)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[1])

    # load weights
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    # model.load_state_dict(torch.load(weights_path)['model'])
    model.to(device)

    # load roi mask
    roi_img = Image.open(roi_mask_path).convert('L')
    roi_img = np.array(roi_img)

    # load image
    original_img = Image.open(img_path).convert('RGB')

    # from pil image to tensor and normalize
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=mean1, std=std1)])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()
    with torch.no_grad():
        # init model
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        output = model(img.to(device))
        t_end = time_synchronized()
        print("inference time: {}".format(t_end - t_start))

        output[output >= 0.5] = 1
        output[output < 0.5] = 0
        # prediction = output.argmax(1).squeeze(0)
        prediction = output.squeeze(0).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        # prediction = output.to("cpu").numpy().astype(np.uint8)
        prediction[prediction == 1] = 255
        # prediction[roi_img == 0] = 0
        mask = Image.fromarray(prediction)
        mask.save("test_result.png")


if __name__ == '__main__':
    main()
