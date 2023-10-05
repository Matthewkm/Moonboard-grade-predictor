#simply return a resnet - not pretrained with an output of 

#as the problem is quite simple, we use a simple model without too many parameters.

import torch.nn as nn

def get_model():

	input_size = 11*18
	output_size = 1 

	layers = []
	
	layers.append(nn.Linear(input_size,612))
	layers.append(nn.ReLU())
	layers.append(nn.BatchNorm1d(612))
	layers.append(nn.Linear(612,1028))
	layers.append(nn.ReLU())
	layers.append(nn.BatchNorm1d(1028))
	layers.append(nn.Linear(1028,2048))
	layers.append(nn.ReLU())
	layers.append(nn.BatchNorm1d(2048))
	layers.append(nn.Linear(2048,2048))
	layers.append(nn.ReLU())
	layers.append(nn.BatchNorm1d(2048))
	layers.append(nn.Linear(2048,2048))
	layers.append(nn.ReLU())
	layers.append(nn.BatchNorm1d(2048))
	layers.append(nn.Linear(2048,1028))
	layers.append(nn.ReLU())
	layers.append(nn.BatchNorm1d(1028))
	layers.append(nn.Linear(1028,612))
	layers.append(nn.ReLU())
	layers.append(nn.BatchNorm1d(612))
	layers.append(nn.Dropout(p=0.8))
	layers.append(nn.Linear(612,output_size))

	return nn.Sequential(*layers)