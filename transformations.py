import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

class Normalize(object):
    def __call__(self, data):
        image, keypoints = data["image"], data["keypoints"]
        image_copy = np.copy(image)
        keypoints_copy = np.copy(keypoints)

        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)
        image_copy = image_copy/255.0

        keypoints_max = np.max(keypoints_copy)
        keypoints_min = np.min(keypoints_copy)

        keypoints_copy = (keypoints -100.0) /50.0

        return  {'image': image_copy, 'keypoints': keypoints_copy}

class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self,data):
        image, keypoints = data["image"], data["keypoints"]

        h, w = image.shape[0], image.shape[1]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * (h/w), self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * (w/h)
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        new_image = cv2.resize(image, (new_w, new_h))
        new_keypoints = keypoints * [new_w/w, new_h/h]

        return  {'image': new_image, 'keypoints': new_keypoints}

class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if(isinstance(output_size, int)):
            self.output_size =(output_size, output_size)
        else:
            self.output_size = output_size

    def __call__(self, data):
        image, keypoints = data["image"], data["keypoints"]

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        new_image = image[top:top+new_h, left:left+new_w]
        new_keypoints = keypoints - [left, top]

        return {'image': new_image, 'keypoints': new_keypoints}


class ToTensor(object):
    def __call__(self, data):
        image, keypoints = data["image"], data["keypoints"]
        if(len(image.shape) == 2 ):
            image = image.reshape(image.shape[0], image.shape[1], 1)

        image = image.transpose((2,0,1))
        image_tensor = torch.from_numpy(image)
        keypoints_tensor = torch.from_numpy(keypoints)

        return {'image': image_tensor, 'keypoints': keypoints_tensor}