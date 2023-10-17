import cv2
import os
import random
import json

def mosaic2(image, step=16):
    h, w, _ = image.shape
    image2 = cv2.resize(image, (w // step, h // step))
    image3 = cv2.resize(image2, (w, h), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite('test.jpg', image3)


if __name__ == '__main__':
	img = cv2.imread('imagenet/val/n01491361/ILSVRC2012_val_00002969.JPEG', 1)
	temp = mosaic2(img)