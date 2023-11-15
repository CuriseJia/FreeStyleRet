import cv2
import os
import random
import json
from tqdm import tqdm

def mosaic2(image_path, out_path, step=16):
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    image2 = cv2.resize(image, (w // step, h // step))
    image3 = cv2.resize(image2, (w, h), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(out_path, image3)


if __name__ == '__main__':
    root_path = 'DSR/'
    folderlist = os.listdir(root_path+'images')
    if os.path.exists(root_path + 'mosaic/'):
        pass
    else:
        os.mkdir(root_path + 'mosaic/')

    for folder in tqdm(folderlist):
        # os.mkdir(root_path + '/mosaic/{}'.format(folder))
        filelist = os.listdir(root_path + 'images/{}'.format(folder))

        for file in tqdm(filelist):
            image_path = root_path + 'images/{0}/{1}'.format(folder, file)
            mosaic_path = root_path + 'mosaic/{0}/{1}'.format(folder, file)

            mosaic2(image_path, mosaic_path)

    print('finish.')