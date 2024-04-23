import pandas as pd
import os
import shutil
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


Applause_path = "./ApplauseData"
Spectogram_Applause_path = "./Spectogram/Applause"
Spectogram_Random_path = "./Spectogram/Randomaudio1"
checkpoint_path = "./Training/"

batch_size = 64
num_classes = 1
learning_rate = 0.001
num_epochs = 20

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def GetSpectograms(path):
    images = []
    for file in os.listdir(path):
        images.append(image.img_to_array(image.load_img(os.path.join(path, file), target_size=(224, 224, 3))))
    return images

def show_images(images):
    fig, axes = plt.subplots(1, 10, figsize=(20, 20), subplot_kw={'xticks': [], 'yticks': []})

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i] / 255)
    plt.show()
    plt.close(fig)

class ConvNeuralNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 128, kernel_size=3)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3)
        self.fc1 = nn.Linear(128 * 12 * 12, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.relu = nn.ReLU()
        self.num_classes = num_classes

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.relu(self.conv4(x))
        x = self.pool(x)
        x = x.reshape(-1, 128 * 12 * 12)  # Reshape before passing to fully connected layer
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)  # Apply softmax along dimension 1 (class scores dimension)
        return x


x = GetSpectograms(Spectogram_Applause_path)
y = [0] * len(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.3, random_state=0)
# normalise dataset
x_train = np.array(x_train) / 255
x_test = np.array(x_test) / 255

# convert to one-hot encoded vectors
encoder = OneHotEncoder()
y_train_encoded = encoder.fit_transform([[label] for label in y_train]).toarray()
y_test_encoded = encoder.transform([[label] for label in y_test]).toarray()

num_classes = 2
# Create an instance of the model
model = ConvNeuralNet(num_classes)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

train_steps = len(x_train)
test_steps = len(x_test)

# batch_size = 64  # Define the batch size

# Split the training data into batches
x_train_batches = [x_train[i:i+batch_size] for i in range(0, len(x_train), batch_size)]
y_train_batches = [y_train_encoded[i:i+batch_size] for i in range(0, len(y_train_encoded), batch_size)]

# Training loop
for epoch in range(num_epochs):
    for i in range(len(x_train_batches)):
        # Convert batch data to tensors
        targets = torch.tensor(x_train_batches[i], dtype=torch.float).permute(0, 3, 1, 2)  # Change shape and permute dimensions
        labels = torch.tensor(y_train_batches[i], dtype=torch.long)
        
        # Forward pass
        outputs = model(targets)
        
        # Compute loss
        loss = criterion(outputs, torch.argmax(labels, dim=1))  # Convert one-hot encoded labels to class indices
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
