#simply return a resnet - not pretrained with an output of 

#as the problem is quite simple, we use a simple model without too many parameters.

import torch.nn as nn
import torch


# Basic CNN...
class ConvNeuralNet(nn.Module):
	#  Determine what layers and their order in CNN object 
    def __init__(self, num_classes):
        super(ConvNeuralNet, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3,padding=1)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,padding=1)
        self.conv_layer3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,padding=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.conv_layer4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3,padding=1)
        self.conv_layer5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,padding=1)
        self.conv_layer6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3,padding=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(512, 1028)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1028, num_classes)
    
    # Progresses data across layers    
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.relu(out)
        out = self.conv_layer2(out)
        out = self.relu(out)
        out = self.conv_layer3(out)
        out = self.relu(out)
        out = self.max_pool1(out)
        
        out = self.conv_layer4(out)
        out = self.relu(out)
        out = self.conv_layer5(out)
        out = self.relu(out)
        out = self.conv_layer6(out)
        out = self.relu(out)
        out = self.max_pool2(out)
        
        out = self.global_pool(out)  
        out = torch.squeeze(out)
        
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def get_cnn_model():

	input_size = 11*18
	output_size = 1
	return ConvNeuralNet(num_classes = output_size)