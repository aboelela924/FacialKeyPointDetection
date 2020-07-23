import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

from data_processing import FacialKeyPointDataProccessing
from transformations import Normalize, Rescale, RandomCrop, ToTensor

from model import Network


def prepare_data(csv_file, image_folder, batch_size=10):
    trainformations = transforms.Compose([
        Rescale(250),
        RandomCrop(224),
        Normalize(),
        ToTensor()
    ])

    dataset = FacialKeyPointDataProccessing(csv_file, image_folder, transformation=trainformations)

    validation_split = .1
    shuffle_dataset = True
    random_seed = 42

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)

    return train_loader, validation_loader


def train(epochs=5, batch_size=10):
    dev = "cuda:0"
    torch.device(dev)
    train_loader, validation_loader = prepare_data(
        "data/training_frames_keypoints.csv",
        "data/training")

    model = Network()
    model.to(dev)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_val_loss = 10
    for epoch in range(epochs):

        running_loss = 0.0

        for batch_num, train_batch in enumerate(train_loader):
            images = train_batch["image"]
            keypoints = train_batch["keypoints"]

            keypoints = keypoints.view(keypoints.size(0), -1)
            images = images.to(dev)
            keypoints = keypoints.to(dev)
            images = images.type(torch.cuda.FloatTensor)
            keypoints = keypoints.type(torch.cuda.FloatTensor)
            # images = images.type(torch.FloatTensor)
            # keypoints = keypoints.type(torch.FloatTensor)

            optimizer.zero_grad()

            prediction = model.forward(images)
            loss = criterion(prediction, keypoints)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_num % 10 == 9:
                print(f'Epoch: {epoch + 1} in batch {batch_num + 1}, average loss is {loss / (10 * batch_size)}')
                running_loss = 0

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_num, val_batch in enumerate(validation_loader):
                images = train_batch["image"]
                keypoints = train_batch["keypoints"]

                keypoints = keypoints.view(keypoints.size(0), -1)
                images = images.to(dev)
                keypoints = keypoints.to(dev)
                images = images.type(torch.cuda.FloatTensor)
                keypoints = keypoints.type(torch.cuda.FloatTensor)

                prediction = model.forward(images)
                loss = criterion(prediction, keypoints)
                val_loss += loss.item()
                if batch_num % 10 == 9:
                    print(
                        f'Epoch: {epoch + 1} in batch {batch_num + 1}, validation loss is {val_loss / (10 * batch_size)}')
                    if best_val_loss > val_loss:
                        print("in if")
                        if (not os.path.isdir("/content/saved_models")):
                            os.mkdir("/content/saved_models")
                        if os.path.exists("/content/saved_models/keypoints_model_9.pt"):
                            os.remove("/content/saved_models/keypoints_model_9.pt")
                        model_dir = '/content/saved_models/'
                        model_name = 'keypoints_model_9.pt'
                        torch.save(model.state_dict(), model_dir + model_name)
                        best_val_loss = val_loss
                    val_loss = 0

        model.train()
    print("Training Done.")

train(epochs=2)