import torch as tc
import io
import cv2
import os
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from typing import Tuple, List
from google.colab import drive

# Note that I have used google Colab to run this code, so I have used the drive to load the data.
# If you want to run this code on your computer, you will have to change the path to the data.
# Why I have used google Colab? For GPU acceleration, it is much faster than my computer CPU.
# You have the test of speed in the end of the code.
# Enjoy!


drive.mount('/content/drive')
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomAffine(
            degrees=5, translate=(0.2, 0.2), scale=(0.9, 1.1)),
        transforms.ToTensor(),

    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
    ]),
}
# Loading data
data_dir = "/content/drive/MyDrive/Colab Notebooks/fvc2000"
image_datasets = {x: datasets.ImageFolder(os.path.join(
    data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: tc.utils.data.DataLoader(
    image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
device = tc.device("cuda:0" if tc.cuda.is_available() else "cpu")


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.axis(False)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    time.sleep(10)


def criterion(outputs, labels, model):
    # calculate cross-entropy loss
    cross_entropy_loss = tc.nn.functional.cross_entropy(outputs, labels)

    # calculate Frobenius norm of the weight matrix in the last layer
    weights = model.fc.weight
    weight_frobenius_norm = tc.norm(weights, p='fro')

    # calculate final loss
    final_loss = cross_entropy_loss + 0.01 * weight_frobenius_norm

    return final_loss


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with tc.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = tc.max(outputs, 1)
                    loss = criterion(outputs, labels, model)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += tc.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(
        f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with tc.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = tc.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
# part 2 of the code for evaluation I have choosen to use the function from figerprint_classic.py to diversify the code


def read_img_paths(dir: str) -> List[str]:
    """
    Read image file names from given directory and return as a list of str: "dir/name.tif"
    """
    filename_list: List[str] = []

    for filename in os.listdir(dir):
        if filename.endswith('.bmp'):
            img_path = os.path.join(dir, filename)
            if os.path.isfile(img_path):
                filename_list.append(img_path)

    filename_list = [filename.replace('\\', '/') for filename in filename_list]

    return filename_list


if __name__ == '__main__':

    model_ft = models.resnet50(pretrained=True)

    # Modify last layer to output number of classes in your dataset
    num_classes = len(class_names)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)

    # Transfer learning
    model_ft = model_ft.to(device)
    optimizer_ft = tc.optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = tc.optim.lr_scheduler.StepLR(
        optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft,
                           exp_lr_scheduler, num_epochs=25)
    # visualize_model(model_ft)
    # Save trained model
    tc.save(model_ft.state_dict(), 'resnet50_trained.pth')

    # Load the saved model
    dir_final_test = '/content/drive/MyDrive/Colab Notebooks/fvc2000_final_test'
    # Define the architecture of the model
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))

    # Load the saved state dict
    state_dict = tc.load('resnet50_trained.pth')
    model.load_state_dict(state_dict)

    # Set the model to evaluation mode
    model.eval()
    test_image_paths = read_img_paths(dir_final_test)

    # Load test images one by one
    for img_path in test_image_paths:
        # Load the image and preprocess it
        idx = int(img_path.split('/')[6].split('.')[0])
        img = cv2.imread(img_path)
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        img_tensor = transform(img)
        # unsqueeze to form a 4D input batch
        img_tensor = img_tensor.unsqueeze(0)

        # Get model predictions for the input batch
        with tc.no_grad():
            outputs = model(img_tensor)
            _, predicted = tc.max(outputs, 1)

        plt.figure(dpi=50)
        plt.imshow(img, cmap='gray')
        plt.axis(False)
        plt.title(f"Predicted --> {predicted[0]} :::: Truth -->{idx}")
        plt.show()


# Output of the first execution using CPU:
# ----------------------------------------
# Epoch 0/24
# ----------
# train Loss: 1.8604 Acc: 0.3506
# val Loss: 1.0182 Acc: 0.6020

# Epoch 1/24
# ----------
# train Loss: 1.4543 Acc: 0.5225
# val Loss: 1.2174 Acc: 0.6617

# Epoch 2/24
# ----------
# train Loss: 1.1226 Acc: 0.6628
# val Loss: 2.1795 Acc: 0.5522

# Epoch 3/24
# ----------
# train Loss: 0.8534 Acc: 0.7329
# val Loss: 0.6861 Acc: 0.7861

# Epoch 4/24
# ----------
# train Loss: 0.6140 Acc: 0.8230
# val Loss: 0.5324 Acc: 0.8358

# Epoch 5/24
# ----------
# train Loss: 0.6083 Acc: 0.8180
# val Loss: 0.4809 Acc: 0.8806

# Epoch 6/24
# ----------
# train Loss: 0.6447 Acc: 0.8063
# val Loss: 0.6049 Acc: 0.8458

# Epoch 7/24
# ----------
# train Loss: 0.3755 Acc: 0.8998
# val Loss: 0.4310 Acc: 0.8806

# Epoch 8/24
# ----------
# train Loss: 0.3129 Acc: 0.9265
# val Loss: 0.3697 Acc: 0.9055

# Epoch 9/24
# ----------
# train Loss: 0.2360 Acc: 0.9449
# val Loss: 0.3645 Acc: 0.8905

# Epoch 10/24
# ----------
# train Loss: 0.2414 Acc: 0.9399
# val Loss: 0.3773 Acc: 0.8856

# Epoch 11/24
# ----------
# train Loss: 0.2482 Acc: 0.9316
# val Loss: 0.2827 Acc: 0.9204

# Epoch 12/24
# ----------
# train Loss: 0.1965 Acc: 0.9432
# val Loss: 0.2694 Acc: 0.9303

# Epoch 13/24
# ----------
# train Loss: 0.2020 Acc: 0.9549
# val Loss: 0.2836 Acc: 0.9154

# Epoch 14/24
# ----------
# train Loss: 0.2030 Acc: 0.9516
# val Loss: 0.3126 Acc: 0.9104

# Epoch 15/24
# ----------
# train Loss: 0.1647 Acc: 0.9649
# val Loss: 0.2843 Acc: 0.9254

# Epoch 16/24
# ----------
# train Loss: 0.1769 Acc: 0.9616
# val Loss: 0.3238 Acc: 0.9055

# Epoch 17/24
# ----------
# train Loss: 0.1851 Acc: 0.9499
# val Loss: 0.2827 Acc: 0.9254

# Epoch 18/24
# ----------
# train Loss: 0.1418 Acc: 0.9733
# val Loss: 0.2337 Acc: 0.9303

# Epoch 19/24
# ----------
# train Loss: 0.1862 Acc: 0.9599
# val Loss: 0.2778 Acc: 0.9154

# Epoch 20/24
# ----------
# train Loss: 0.2157 Acc: 0.9466
# val Loss: 0.1853 Acc: 0.9502

# Epoch 21/24
# ----------
# train Loss: 0.1893 Acc: 0.9566
# val Loss: 0.3064 Acc: 0.9005

# Epoch 22/24
# ----------
# train Loss: 0.1966 Acc: 0.9549
# val Loss: 0.1850 Acc: 0.9552

# Epoch 23/24
# ----------
# train Loss: 0.1741 Acc: 0.9583
# val Loss: 0.2740 Acc: 0.9104

# Epoch 24/24
# ----------
# train Loss: 0.1708 Acc: 0.9566
# val Loss: 0.2242 Acc: 0.9303

# Training complete in 34m 6s
# Best val Acc: 0.955224

# ----------------------------------------

# Output of the second execution using GPU on Google Colab:
# ----------------------------------------
# Epoch 0/24
# ----------
# train Loss: 1.9299 Acc: 0.3122
# val Loss: 1.0738 Acc: 0.6766

# Epoch 1/24
# ----------
# train Loss: 1.3073 Acc: 0.5810
# val Loss: 1.0117 Acc: 0.6965

# Epoch 2/24
# ----------
# train Loss: 1.0111 Acc: 0.6895
# val Loss: 1.2257 Acc: 0.6418

# Epoch 3/24
# ----------
# train Loss: 0.8714 Acc: 0.7312
# val Loss: 0.8168 Acc: 0.7413

# Epoch 4/24
# ----------
# train Loss: 0.6099 Acc: 0.8097
# val Loss: 0.4489 Acc: 0.9005

# Epoch 5/24
# ----------
# train Loss: 0.4950 Acc: 0.8614
# val Loss: 0.7073 Acc: 0.8010

# Epoch 6/24
# ----------
# train Loss: 0.5051 Acc: 0.8614
# val Loss: 0.9959 Acc: 0.8109

# Epoch 7/24
# ----------
# train Loss: 0.3739 Acc: 0.8932
# val Loss: 0.5718 Acc: 0.8806

# Epoch 8/24
# ----------
# train Loss: 0.2644 Acc: 0.9316
# val Loss: 0.4573 Acc: 0.8806

# Epoch 9/24
# ----------
# train Loss: 0.2166 Acc: 0.9466
# val Loss: 0.3111 Acc: 0.9254

# Epoch 10/24
# ----------
# train Loss: 0.2789 Acc: 0.9249
# val Loss: 0.3793 Acc: 0.8905

# Epoch 11/24
# ----------
# train Loss: 0.2532 Acc: 0.9349
# val Loss: 0.2446 Acc: 0.9403

# Epoch 12/24
# ----------
# train Loss: 0.2635 Acc: 0.9449
# val Loss: 0.3682 Acc: 0.9005

# Epoch 13/24
# ----------
# train Loss: 0.2119 Acc: 0.9499
# val Loss: 0.3457 Acc: 0.8905

# Epoch 14/24
# ----------
# train Loss: 0.1705 Acc: 0.9549
# val Loss: 0.3152 Acc: 0.9005

# Epoch 15/24
# ----------
# train Loss: 0.1919 Acc: 0.9499
# val Loss: 0.2309 Acc: 0.9353

# Epoch 16/24
# ----------
# train Loss: 0.1856 Acc: 0.9516
# val Loss: 0.3019 Acc: 0.8955

# Epoch 17/24
# ----------
# train Loss: 0.1594 Acc: 0.9633
# val Loss: 0.1771 Acc: 0.9453

# Epoch 18/24
# ----------
# train Loss: 0.2169 Acc: 0.9482
# val Loss: 0.3030 Acc: 0.9055

# Epoch 19/24
# ----------
# train Loss: 0.2002 Acc: 0.9516
# val Loss: 0.2128 Acc: 0.9303

# Epoch 20/24
# ----------
# train Loss: 0.1493 Acc: 0.9750
# val Loss: 0.1704 Acc: 0.9403

# Epoch 21/24
# ----------
# train Loss: 0.1769 Acc: 0.9583
# val Loss: 0.3244 Acc: 0.8905

# Epoch 22/24
# ----------
# train Loss: 0.1611 Acc: 0.9566
# val Loss: 0.1910 Acc: 0.9353

# Epoch 23/24
# ----------
# train Loss: 0.1451 Acc: 0.9683
# val Loss: 0.1692 Acc: 0.9403

# Epoch 24/24
# ----------
# train Loss: 0.1815 Acc: 0.9616
# val Loss: 0.2609 Acc: 0.8955

# Training complete in 4m 11s
# Best val Acc: 0.945274
