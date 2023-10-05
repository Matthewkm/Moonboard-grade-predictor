import pandas as pd
from ast import literal_eval
import numpy as np
import torch

class MoonboardDataset():

	def __init__(self,problem_file,flatten=False,test=False,full_dataset=False):
		#open the folder.
		data = pd.read_csv(problem_file)
		#remove any 6B problems
		data = data[data['grade']!='6B']
		#remove 8A+/B/B+ as either not enough good quality non benchmarks to train on and vast majority become meme problems...
		data = data[data['grade']!='8B+']
		data = data[data['grade']!='8B']
		data = data[data['grade']!='8A+']

		#we now perform some filtering to remove meme and garbage...
		data = data[data['repeats']>20]
		#print(data['userRating'])
		data = data[data['grade']==data['userGrade']]

		if full_dataset:
			data = data #return full dataset with both bechmarks and non benchmarks
		elif test==True:
			data = data[data['isBenchmark']==True]
		else:
			data = data[data['isBenchmark']==False]

		self.holds = data['moves'].values
		self.grades = data['grade'].values
		#filder the problems to ensure we have only have ones that are non meme 

		self.all_columns = np.asarray(['A','B','C','D','E','F','G','H','I','J','K'])
		self.all_grades = np.asarray(['6B+','6C','6C+','7A','7A+','7B','7B+','7C','7C+','8A','8A+','8B'])

		self.flatten = flatten

	def __len__(self):
		return len(self.grades)

	def __getitem__(self,idx):
		#convert holds to lists as they are stored as strings...

		grade = np.asarray(np.argwhere(self.grades[idx]==self.all_grades)[0][0])
		grade = torch.from_numpy(grade).float()

		moves = literal_eval(self.holds[idx]) #list of holds

		move_map = np.zeros((18,11))
		for move in moves: #may be a way to do this w/o a loop

			column = np.argwhere(move[0].capitalize()==self.all_columns)[0][0]
			row = move[1:]

			move_map[18-(int(row)),int(column)] = 1

		#convert to pytorch tensor and flatten...
		move_map = torch.from_numpy(move_map)
		if self.flatten:
			move_map = torch.flatten(move_map)
		else:
			move_map = torch.unsqueeze(move_map,dim=0)

		return move_map,grade