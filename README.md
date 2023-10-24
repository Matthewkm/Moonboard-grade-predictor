# Moonboard Grade Prediction with Deep Learning

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
| Metric       | CNN (ours) | Neural Network (ours) | RNN [[Duh & Chang]](https://arxiv.org/pdf/2102.01788.pdf) |
| ------------ | ---------- | --------------------- | --------------------------------------------------------- |
| Accuracy     |   59.4%    |   **71.4%**           |        46.7%                                             | 
| Accuracy +/-1|   71.4%    |   **95.9%**           |        84.7%

- We provide pretrained weights for our custom Neural and Convolutional Neural Networks for grade prediction.
  - These can be downloaded from my google drive as they are too big for github: [NN weights](https://drive.google.com/file/d/1HFXFQCYpmgARNR_Hz6o-I2MM4x5toEWF/view?usp=sharing), [CNN weights](https://drive.google.com/file/d/1Latig7ldjil_XG9PhQW6an2mVFuy-6gd/view?usp=sharing) or you can train your own models utilising the training code.
- We provide a Jupyter notebook `Testing.ipynb` for analysing results output by our model, this will require pytorch to reproduce on your machine.
- The Neural Network performs significantly better than the CNN (71.4% vs 59.4% closest grade prediction and 95.9% vs 80.1% +/- 1 grade), potenitally due to increased size, but also max pooling probably hurts the CNNs ability to model exact hold relations.

## Retraining the models
If you wish to train your own models run ```trainer.py```. for convenience we include two arguments ```--model_type``` which can be set to 'NN' or 'CNN' and ```benchmark``` which can be set to True for training on non benchmark problems rather than a random split.

# Notes
This code is a work in progress and not a finished project, it will be cleaned up and added to over time, however I have made it public at request.
