from models.NN_model import get_model
from models.CNN_model import get_cnn_model
from dataset import MoonboardDataset
import torch
from torch.utils.data import DataLoader
from utils import AverageMeter, accuracy, accuracy_regression
import argparse

#load in the model...

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', help='The type of model utilise',default='NN')
parser.add_argument('--benchmark',help='Set to True to train on non benchmarks and test on benchmarks',type=bool,default=False)

args = parser.parse_args()

assert args.model_type in ['NN','CNN']

if args.model_type == 'NN':
	model = get_model()
	flatten = True
elif args.model_type == 'CNN':
	model = get_cnn_model()
	flatten = False

if args.benchmark:
	train_dataset = MoonboardDataset('2016_moonboard_problems.csv',flatten=flatten)
	test_dataset = MoonboardDataset('2016_moonboard_problems.csv',flatten=flatten,test=True)
else:
	print('Utilising random 80/20 split')
	dataset = MoonboardDataset('2016_moonboard_problems.csv',full_dataset=True,flatten=flatten)
	train_size = int(0.8 * len(dataset))
	test_size = len(dataset) - train_size
	generator = torch.Generator().manual_seed(1337)
	train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=generator)


#get the data loaders. 
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)


loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)


def train_epoch():

	print('Running training epoch')
	losses = AverageMeter()
	top1 = AverageMeter()
	top3 = AverageMeter()

	for i,data in enumerate(train_dataloader):

		inputs,labels = data

		optimizer.zero_grad()

		outputs = model(inputs.float())
		outputs = outputs.squeeze()
		loss = loss_fn(outputs,labels)

		prec1,prec3 = accuracy_regression(outputs.data, labels)
		losses.update(loss.item(), inputs.size(0))
		top1.update(prec1.item(), inputs.size(0))
		top3.update(prec3.item(), inputs.size(0))

		loss.backward()

		optimizer.step()
		optimizer.zero_grad()

		if i % 20 == 0:
			output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
						'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
						'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
 						'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
				epoch, i, len(train_dataloader), loss=losses, top1=top1, top3=top3, lr=optimizer.param_groups[-1]['lr']*0.1))
			print(output)


def val_epoch():

	print('Running validation epoch')

	losses = AverageMeter()
	top1 = AverageMeter()
	top3 = AverageMeter()
	model.eval()

	with torch.no_grad():
		for i,data in enumerate(test_dataloader):

			inputs,labels = data

			outputs = model(inputs.float())
			outputs = outputs.squeeze()
			loss = loss_fn(outputs,labels)

			prec1,prec3 = accuracy_regression(outputs.data, labels)

			losses.update(loss.item(), inputs.size(0))
			top1.update(prec1.item(), inputs.size(0))
			top3.update(prec3.item(), inputs.size(0))

		output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
					'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
						'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
			epoch, i, len(test_dataloader), loss=losses, top1=top1, top3=top3, lr=optimizer.param_groups[-1]['lr']*0.1))
		print(output)


	return top1.avg

best_top1 = 0
for epoch in range(100):
	train_epoch()
	curr_top1 = val_epoch()

	print('current top1 is: {}, best is {}.'.format(curr_top1,best_top1))
	if curr_top1 > best_top1:
		torch.save(model.state_dict(),'grade_predictor.pt')
		print('saved model')
		best_top1 = curr_top1