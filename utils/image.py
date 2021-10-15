import numpy as np

def normalize(img):
    img = img.astype(np.float32)
    if img.max() == img.min():
        return img
    else:
        return (img - np.min(img)) / (np.max(img) - np.min(img))