# Moonboard grade predictor

This is a work in progress minimal implementation of a moonboard grade predictor model utilising a custom Neural and Convolutional Neural architecture.

All models are implemented within the Pytorch deep learning frame work, instructions on download pytorch can be found on their [webside](https://pytorch.org/).
Please note cuda is not required within this work due to the very small scale datasets and models utilised.

## Dataset
We provide a scaled down .csv version of the moonboard 2016 dataset, utilised to train and test the models.
This dataset was downloaded via https://github.com/spookykat/MoonBoard as of 30/07/2023 and contains 65890 problems.

## Method
- [documentation to be Finished/Improved at some point , untill then you'll have to work your way through the code.]
  - We utilise a 11x18 grid to represent the holds within a moon board. For any given climb the holds utilised will be set to 1, with the rest remaining 0.
  - We train the model either utilising a random 80:20 split (default) or train on non bench marks and test on benchmarks by setting ```--benchmark True``` during training.
    - In both cases, we filter all problems with less than 20 logs, and where the user grade does not match the setters grade.
  - The grid is then utilised as the input to the CNN model (or flattened for input to the neural network)
  - We found performing regression with a Mean Squared error performed best.

## Results
We provide pretrained weights for our custom Neural and Convolutional Neural Networks for grade prediction.
We provide a Jupyter notebook `Testing.ipynb` for analysing results output by our model, this will require pytorch to reproduce on your machine.
The ranalysis will be improved in the future.

## Retraining the models
If you wish to train your own models run ```trainer.py```. for convenience we include two arguments ```--model_type``` which can be set to 'NN' or 'CNN' and ```benchmark``` which can be set to True for training on non benchmark problems rather than a random split.

# Notes
This code is a work in progress and not a finished project, it will be cleaned up and added to over time, however I have made it public at request.
