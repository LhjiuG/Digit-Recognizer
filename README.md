# Digit-Recognizer
Hand written digits recognizer using a convolutional neural network.

This was made for the kaggle ["competition"](https://www.kaggle.com/c/digit-recognizer). In the "competition", we had to create a network that could accurately label handwritten digits. There are multiple way to do this problem and I decided to go with a convolutional neural network. The architecture used is based on Lenet-5 with some ameliorations. 

## What I have learned?
* Using data from the internet.
* Normalization / One hot encoding 
* Using the Keras functionnal api to rapidly make and use a deep neural network.
* Data plotting.

## What problems did I encounter?
The main problems I have encountered was the big variance between the training set (~98% accuracy) and the test set (below 50% accuracy).

### How I fixed it?
The issues came from the was the data was initialized. After adding some data preprocessing scheme and dropout layers, I was able to fix the overheating during training.

## Results
I was able to achieve an accuracy of 99.17% on the validation test.

