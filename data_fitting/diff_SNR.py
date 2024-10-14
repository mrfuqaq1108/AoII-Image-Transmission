import torch, torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

# Applying Transforms to the Data
image_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

# Load the Data

# Set train and valid directory paths

dataset = 'raw-img'

train_directory = os.path.join(dataset, 'train')
valid_directory = os.path.join(dataset, 'valid')
test_directory = os.path.join(dataset, 'test')

# Batch size
bs = 32

# Number of classes
num_classes = len(os.listdir(valid_directory))  # 10
print(num_classes)

# Load Data from folders
data = {
    'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
    'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid']),
    'test': datasets.ImageFolder(root=test_directory, transform=image_transforms['test'])
}

# Get a mapping of the indices to the class names, in order to see the output classes of the test images.
idx_to_class = {v: k for k, v in data['train'].class_to_idx.items()}
print(idx_to_class)

# Size of Data, to be used for calculating Average Loss and Accuracy
train_data_size = len(data['train'])
valid_data_size = len(data['valid'])
test_data_size = len(data['test'])

# Create iterators for the Data loaded using DataLoader module
train_data_loader = DataLoader(data['train'], batch_size=bs, shuffle=True)
valid_data_loader = DataLoader(data['valid'], batch_size=bs, shuffle=True)
test_data_loader = DataLoader(data['test'], batch_size=bs, shuffle=True)


# Load pretrained ResNet50 Model
resnet50 = models.resnet50(pretrained=True)
# resnet50 = resnet50.to('cuda:0')

# Freeze model parameters
for param in resnet50.parameters():
    param.requires_grad = False


# Change the final layer of ResNet50 Model for Transfer Learning
fc_inputs = resnet50.fc.in_features

resnet50.fc = nn.Sequential(
    nn.Linear(fc_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 10),
    nn.LogSoftmax(dim=1))

loss_func = nn.NLLLoss()
optimizer = optim.Adam(resnet50.parameters())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_and_validate(model, loss_criterion, optimizer, epochs=25):
    '''
    Function to train and validate
    Parameters
        :param model: Model to train and validate
        :param loss_criterion: Loss Criterion to minimize
        :param optimizer: Optimizer for computing gradients
        :param epochs: Number of epochs (default=25)

    Returns
        model: Trained Model with best validation accuracy
        history: (dict object): Having training loss, accuracy and validation loss, accuracy
    '''

    start = time.time()
    history = []
    best_acc = 0.0

    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch + 1, epochs))

        # Set to training mode
        model.train()

        # Loss and Accuracy within the epoch
        train_loss = 0.0
        train_acc = 0.0

        valid_loss = 0.0
        valid_acc = 0.0

        for i, (inputs, labels) in enumerate(train_data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Clean existing gradients
            optimizer.zero_grad()

            # Forward pass - compute outputs on input data using the model
            outputs = model(inputs)

            # Compute loss
            loss = loss_criterion(outputs, labels)

            # Backpropagate the gradients
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Compute the total loss for the batch and add it to train_loss
            train_loss += loss.item() * inputs.size(0)

            # Compute the accuracy
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            # Compute total accuracy in the whole batch and add to train_acc
            train_acc += acc.item() * inputs.size(0)

            # print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))

        # Validation - No gradient tracking needed
        with torch.no_grad():

            # Set to evaluation mode
            model.eval()

            # Validation loop
            for j, (inputs, labels) in enumerate(valid_data_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass - compute outputs on input data using the model
                outputs = model(inputs)

                # Compute loss
                loss = loss_criterion(outputs, labels)

                # Compute the total loss for the batch and add it to valid_loss
                valid_loss += loss.item() * inputs.size(0)

                # Calculate validation accuracy
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                # Convert correct_counts to float and then compute the mean
                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                # Compute total accuracy in the whole batch and add to valid_acc
                valid_acc += acc.item() * inputs.size(0)

                # print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))

        # Find average training loss and training accuracy
        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size

        # Find average training loss and training accuracy
        avg_valid_loss = valid_loss / valid_data_size
        avg_valid_acc = valid_acc / valid_data_size

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        epoch_end = time.time()

        print(
            "Epoch : {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation : Loss : {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
                epoch, avg_train_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100,
                epoch_end - epoch_start))

        # Save if the model has best accuracy till now
        torch.save(model, dataset + '_model_' + str(epoch) + '.pt')

    return model, history


num_epochs = 30
# trained_model, history = train_and_validate(resnet50, loss_func, optimizer, num_epochs)

# torch.save(history, dataset + '_history.pt')


# loss picture
'''
history = torch.load('raw-img_history.pt')

history = np.array(history)
plt.plot(history[:, 0:2])
plt.legend(['Tr Loss', 'Val Loss'])
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.ylim(0, 1)
plt.savefig(dataset + '_loss_curve.png')
plt.show()
plt.plot(history[:, 2:4])
plt.legend(['Tr Accuracy', 'Val Accuracy'])
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.savefig(dataset + '_accuracy_curve.png')
plt.show()
'''


def computeTestSetAccuracy(model, loss_criterion):
    '''
    Function to compute the accuracy on the test set
    Parameters
        :param model: Model to test
        :param loss_criterion: Loss Criterion to minimize
    '''

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_acc = 0.0
    test_loss = 0.0

    # Validation - No gradient tracking needed
    with torch.no_grad():
        # Set to evaluation mode
        model.eval()

        # Validation loop
        for j, (inputs, labels) in enumerate(test_data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass - compute outputs on input data using the model
            outputs = model(inputs)

            # Compute loss
            loss = loss_criterion(outputs, labels)

            # Compute the total loss for the batch and add it to valid_loss
            test_loss += loss.item() * inputs.size(0)

            # Calculate validation accuracy
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            # Compute total accuracy in the whole batch and add to valid_acc
            test_acc += acc.item() * inputs.size(0)

            print("Test Batch number: {:03d}, Test: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))

    # Find average test loss and test accuracy
    avg_test_loss = test_loss / test_data_size
    avg_test_acc = test_acc / test_data_size

    print("Test accuracy : " + str(avg_test_acc))


'''
# Epoch: 1/30
# Epoch : 000, Training: Loss: 0.4081, Accuracy: 87.2875%,
# 		Validation : Loss : 0.1783, Accuracy: 95.0667%, Time: 687.6961s
# Epoch: 2/30
# Epoch : 001, Training: Loss: 0.2158, Accuracy: 92.9250%,
# 		Validation : Loss : 0.1744, Accuracy: 94.8667%, Time: 696.9382s
# Epoch: 3/30
# Epoch : 002, Training: Loss: 0.1933, Accuracy: 93.9500%,
# 		Validation : Loss : 0.1695, Accuracy: 95.1333%, Time: 673.2229s
# Epoch: 4/30
# Epoch : 003, Training: Loss: 0.1752, Accuracy: 94.3125%,
# 		Validation : Loss : 0.1675, Accuracy: 95.0000%, Time: 671.2522s
# Epoch: 5/30
# Epoch : 004, Training: Loss: 0.1777, Accuracy: 94.0875%,
# 		Validation : Loss : 0.1851, Accuracy: 94.7333%, Time: 672.6484s
# Epoch: 6/30
# Epoch : 005, Training: Loss: 0.1679, Accuracy: 94.5875%,
# 		Validation : Loss : 0.1821, Accuracy: 94.7667%, Time: 671.4825s
# Epoch: 7/30
# Epoch : 006, Training: Loss: 0.1650, Accuracy: 94.7125%,
# 		Validation : Loss : 0.1884, Accuracy: 94.7000%, Time: 670.7305s
# Epoch: 8/30
# Epoch : 007, Training: Loss: 0.1487, Accuracy: 95.0375%,
# 		Validation : Loss : 0.2022, Accuracy: 94.5333%, Time: 671.5035s
# Epoch: 9/30
# Epoch : 008, Training: Loss: 0.1663, Accuracy: 94.7875%,
# 		Validation : Loss : 0.1587, Accuracy: 95.4667%, Time: 673.6408s
# Epoch: 10/30
# Epoch : 009, Training: Loss: 0.1457, Accuracy: 95.3250%,
# 		Validation : Loss : 0.1752, Accuracy: 95.0667%, Time: 672.0351s
# Epoch: 11/30
# Epoch : 010, Training: Loss: 0.1545, Accuracy: 94.9625%,
# 		Validation : Loss : 0.1858, Accuracy: 95.1667%, Time: 672.0779s
# Epoch: 12/30
# Epoch : 011, Training: Loss: 0.1454, Accuracy: 95.3625%,
# 		Validation : Loss : 0.1829, Accuracy: 94.9000%, Time: 671.1404s
# Epoch: 13/30
# Epoch : 012, Training: Loss: 0.1496, Accuracy: 95.0875%,
# 		Validation : Loss : 0.2004, Accuracy: 94.6667%, Time: 679.3117s
# Epoch: 14/30
# Epoch : 013, Training: Loss: 0.1467, Accuracy: 95.3625%,
# 		Validation : Loss : 0.1762, Accuracy: 95.2333%, Time: 683.0674s
# Epoch: 15/30
# Epoch : 014, Training: Loss: 0.1305, Accuracy: 95.7875%,
# 		Validation : Loss : 0.1856, Accuracy: 95.4000%, Time: 679.2758s
# Epoch: 16/30
# Epoch : 015, Training: Loss: 0.1408, Accuracy: 95.5875%,
# 		Validation : Loss : 0.1774, Accuracy: 95.2000%, Time: 682.3467s
# Epoch: 17/30
# Epoch : 016, Training: Loss: 0.1497, Accuracy: 94.9375%,
# 		Validation : Loss : 0.1744, Accuracy: 95.0000%, Time: 680.3578s
# Epoch: 18/30
# Epoch : 017, Training: Loss: 0.1253, Accuracy: 95.6625%,
# 		Validation : Loss : 0.1791, Accuracy: 95.1000%, Time: 681.0669s
# Epoch: 19/30
# Epoch : 018, Training: Loss: 0.1319, Accuracy: 95.7250%,
# 		Validation : Loss : 0.1908, Accuracy: 94.9333%, Time: 681.7600s
# Epoch: 20/30
# Epoch : 019, Training: Loss: 0.1294, Accuracy: 95.7375%,
# 		Validation : Loss : 0.1815, Accuracy: 94.9333%, Time: 682.2547s
# Epoch: 21/30
# Epoch : 020, Training: Loss: 0.1369, Accuracy: 95.6750%,
# 		Validation : Loss : 0.1947, Accuracy: 95.0667%, Time: 682.0812s
# Epoch: 22/30
# Epoch : 021, Training: Loss: 0.1273, Accuracy: 95.8750%,
# 		Validation : Loss : 0.1905, Accuracy: 95.2000%, Time: 680.6480s
# Epoch: 23/30
# Epoch : 022, Training: Loss: 0.1226, Accuracy: 96.0500%,
# 		Validation : Loss : 0.1781, Accuracy: 95.3000%, Time: 681.0190s
# Epoch: 24/30
# Epoch : 023, Training: Loss: 0.1298, Accuracy: 95.7375%,
# 		Validation : Loss : 0.1788, Accuracy: 95.3333%, Time: 681.2703s
# Epoch: 25/30
# Epoch : 024, Training: Loss: 0.1202, Accuracy: 95.9000%,
# 		Validation : Loss : 0.1701, Accuracy: 95.7667%, Time: 681.0310s
# Epoch: 26/30
# Epoch : 025, Training: Loss: 0.1292, Accuracy: 95.9750%,
# 		Validation : Loss : 0.1779, Accuracy: 95.4333%, Time: 681.0978s
# Epoch: 27/30
# Epoch : 026, Training: Loss: 0.1290, Accuracy: 95.8750%,
# 		Validation : Loss : 0.1742, Accuracy: 96.0333%, Time: 680.8525s
# Epoch: 28/30
# Epoch : 027, Training: Loss: 0.1270, Accuracy: 95.8500%,
# 		Validation : Loss : 0.1781, Accuracy: 95.5667%, Time: 680.9831s
# Epoch: 29/30
# Epoch : 028, Training: Loss: 0.1213, Accuracy: 96.2250%,
# 		Validation : Loss : 0.1977, Accuracy: 95.1000%, Time: 680.8754s
# Epoch: 30/30
# Epoch : 029, Training: Loss: 0.1210, Accuracy: 96.1000%,
# 		Validation : Loss : 0.1933, Accuracy: 95.5000%, Time: 680.9083s
#
# 进程已结束,退出代码0
'''


def predict(model, test_image_name):
    '''
    Function to predict the class of a single test image
    Parameters
        :param model: Model to test
        :param test_image_name: Test image
    '''

    transform = image_transforms['test']

    test_image = Image.open(test_image_name)
    plt.imshow(test_image)

    test_image_tensor = transform(test_image)

    if torch.cuda.is_available():
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224).cuda()
    else:
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224)

    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        start = time.time()
        out = model(test_image_tensor)
        stop = time.time()
#        print('cost time', stop - start)
    ps = torch.exp(out)
    topk, topclass = ps.topk(3, dim=1)
    # for i in range(3):
    #     print("Predcition", i + 1, ":", idx_to_class[topclass.cpu().numpy()[0][i]], ", Score: ",
    #           topk.cpu().numpy()[0][i])
    return idx_to_class[topclass.cpu().numpy()[0][0]]


model = torch.load('raw-img_model_5.pt')
# predict(model, 'raw-img/test/cat/1100.jpeg')
# print(predict(model, 'raw-img/test/cat/1100.jpeg'))
# # predict(model, 'caltech_10/test111/11change0.8.jpg')

counter = 0
total = 0
for filename in os.listdir('diff_SNR/test90'):
    # for file in os.path.join('raw-img/test', filename):
    #     print(file)
    # print(filename)
    # print('type =', type(filename))
    a = os.path.join('diff_SNR/test90', filename)
    for file in os.listdir(a):
        total += 1
        print('total =', total)
        sp_file_directory = os.path.join(a, file)
        # print('predict = ', predict(model, sp_file_directory))
        # print('type = ', type(predict(model, sp_file_directory)))
        b = predict(model, sp_file_directory)
        if filename != b:
            counter += 1
            print('counter = ', counter)
        print('percentage = ', (1-counter/total))

# print('counter = ', counter)
# print('total =', total)


'''
100% SNR
counter = 578
total = 15179
0.9619

test10
90% SNR
counter = 1820
total = 15179
0.8801

test20
80% SNR
counter = 3176
total = 15179
0.7908

test30
70% SNR
counter = 6103
total = 15179
0.5979

test40
60% SNR
counter = 9795
total = 15179
0.3547

test50
50% SNR
counter = 12452
total = 15179
0.1797

test60
40% SNR
counter = 13586
total = 15179
0.1049

test70
30% SNR
counter = 14120
total = 15179
0.0698

test80
20% SNR
counter = 14389
total = 15179
0.0520

test90
10% SNR
counter = 14446
total = 15179
0.0483
'''


