import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from model import Network
from transformations import Rescale, RandomCrop, Normalize, ToTensor

image = cv2.imread("images/MrsMaisel.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.show()

face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(image, 1.5, 2)
image_with_detections = image.copy()
for (x,y,w,h) in faces:
    cv2.rectangle(image_with_detections, (x, y), (x + w, y + h), (255, 0, 0), 3)
fig = plt.figure(figsize=(9,9))

plt.imshow(image_with_detections)
plt.show()

net = Network()
net.load_state_dict(torch.load('saved_models/keypoints_model_9.pt'))
net.eval()

image_copy = np.copy(image)

i = 0
for (x, y, w, h) in faces:
    roi = image_copy[y:y + h, x:x + w]
    roi_temp = roi
    roi_temp = cv2.resize(roi_temp,(224,224))
    roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    roi = roi / 255.0
    roi = cv2.resize(roi, (224, 224))
    roi = roi[:, :, None]
    roi = roi.transpose((2, 0, 1))
    roi = torch.from_numpy(roi)
    roi = roi[None, :, :, :]
    result = net(roi.float())
    result = result.view(68, -1)
    result = result.detach()
    result = result * 50.0 + 100.0
    plt.imshow(np.squeeze(roi), cmap="gray")
    plt.scatter(result[:, 0], result[:, 1], s=25, marker='.', c='m')
    plt.show()
