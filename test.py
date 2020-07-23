import torch
from model import Network
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from data_processing import FacialKeyPointDataProccessing
from transformations import Normalize, RandomCrop, Rescale, ToTensor
from utils import net_sample_output, denormalize_keypoints, visualize_output

net = Network()
net.load_state_dict(torch.load("saved_models/keypoints_model_9.pt"))
net.eval()
transformation = transforms.Compose([
    Rescale(250),
    RandomCrop(224),
    Normalize(),
    ToTensor()])
test_dataset = FacialKeyPointDataProccessing("data/test_frames_keypoints.csv",
                                             "data/test",
                                             transformation=transformation)

batch_size = 10
test_loader = DataLoader(test_dataset,
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=0)


images, output_pts, key_pts = net_sample_output(test_loader, net)
# output_pts = output_pts.detach().numpy()
# output_pts = denormalize_keypoints(output_pts)
# key_pts = denormalize_keypoints(key_pts)
visualize_output(images, output_pts)