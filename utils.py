import matplotlib.pyplot as plt
import torch
import numpy as np

def show_key_points(image, keypoints, pred_keypoints=None):
    plt.imshow(np.squeeze(image), cmap="gray")
    plt.scatter(keypoints[:, 0], keypoints[:, 1], s=20, marker='.', c='m')
    if pred_keypoints is not None:
        plt.scatter(pred_keypoints[:, 0], pred_keypoints[:, 1], s=20, marker='.', c='g')

def show_multiple_images(images, keypoints):
    fig = plt.figure()
    for i, image in enumerate(images):
        ax = fig.add_subplot(3,4,i+1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(image), cmap="gray")
        ax.scatter(keypoints[:, 0], keypoints[:, 1], s=20, marker='.', c='m')


def net_sample_output(test_loader, net):
    for i, sample in enumerate(test_loader):

        images = sample['image']
        key_pts = sample['keypoints']

        images = images.type(torch.FloatTensor)

        output_pts = net(images)

        output_pts = output_pts.view(output_pts.size()[0], 68, -1)

        if i == 0:
            return images, output_pts, key_pts

def denormalize_keypoints(keypoints):
    return ((keypoints*50)+100)

def show_all_keypoints(image, predicted_key_pts, gt_pts=None):
    """Show image with predicted keypoints"""
    # image is grayscale
    plt.imshow(image, cmap='gray')
    plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=20, marker='.', c='m')
    # plot ground truth points as green pts
    if gt_pts is not None:
        plt.scatter(gt_pts[:, 0], gt_pts[:, 1], s=20, marker='.', c='g')
def visualize_output(test_images, test_outputs, gt_pts=None, batch_size=10):
    for i in range(batch_size):
        plt.figure(figsize=(20, 10))
        ax = plt.subplot(1, batch_size, i + 1)

        # un-transform the image data
        image = test_images[i].data  # get the image from it's wrapper
        image = image.numpy()  # convert to numpy array from a Tensor
        image = np.transpose(image, (1, 2, 0))  # transpose to go from torch to numpy image

        # un-transform the predicted key_pts data
        predicted_key_pts = test_outputs[i].data
        predicted_key_pts = predicted_key_pts.numpy()
        # undo normalization of keypoints
        predicted_key_pts = predicted_key_pts * 50.0 + 100

        # plot ground truth points for comparison, if they exist
        ground_truth_pts = None
        if gt_pts is not None:
            ground_truth_pts = gt_pts[i]
            ground_truth_pts = ground_truth_pts * 50.0 + 100

        # call show_all_keypoints
        show_all_keypoints(np.squeeze(image), predicted_key_pts, ground_truth_pts)

        plt.axis('off')

    plt.show()
