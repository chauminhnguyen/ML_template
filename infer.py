import torch
import os
from data import FolderDataset, ImageFolderDataset
import cv2


def test(model, data_path, transform=None):
    if os.path.isdir(data_path):
        print("Testing on a folder")
        dataset = ImageFolderDataset(data_path, batch_size=2, transform=transform)
        test_on_batch(model, dataset)
    else:
        print("Testing on a single image")
        img = cv2.imread(data_path)
        if transform is not None:
            img = transform(img)
        img = img.unsqueeze(0)
        test_on_sample(model, img)

def test_on_sample(model, sample):
    model.eval()
    with torch.no_grad():
        output = model(sample)
        print(output)

def test_on_batch(model, batch):
    model.eval()
    with torch.no_grad():
        print(len(batch))
        for sample, cls in batch:
            output = model(sample)
            print(output)