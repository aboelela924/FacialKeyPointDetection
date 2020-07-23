import torch
import matplotlib.pyplot as plt
import cv2

from model import Network

def visualize_filter(weights):
    plt.imshow(weights, cmap="gray")
    plt.show()


def apply_filter(filter, image_dir="images/shaileneWoodley.png"):
    img = cv2.imread(image_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(img, cmap="gray")
    plt.title("Before filter")
    plt.show()
    filtered = cv2.filter2D(img, -1, filter)
    plt.imshow(filtered, cmap="gray")
    plt.title("After filter")
    plt.show()

net = Network()
net.load_state_dict(torch.load("saved_models/keypoints_model_9.pt"))
first_layer_filters = net.conv1.weight.data
first_layer_filters = first_layer_filters.numpy()

visualize_filter(first_layer_filters[25][0])
apply_filter(first_layer_filters[25][0])