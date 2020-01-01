# Text classification

## Objective
Given the consumer complaint narrative, the task is to predict a category that the complaint is about.

## Motivation
This is a multi-class text classification problem. Text classification can have a lot of practical uses. For example, sentiment analysis to help companies track how their brand is perceived online, routing support tickets to the right team based on the type of issue that the customer comment is about (like this project), classifying posts on social media to help social media companies track the topics a certain user is interested in for advertising purposes, etc.

For all text classification tasks using LSTM, no matter what dataset we are working on, a biology dataset, a finance dataset, or social media posts,  the models use the same algorithm and have the same interface. So the code presented in this project can be easily applied to other use cases. The only thing we need to do is reframe the problem and figure out the output categories that the information extracted from the text can be mapped into.

## Descriptions
Dataset and pretrained work embeddings can be found here.
[Dataset](https://www.kaggle.com/cfpb/us-consumer-finance-complaints) |
[Pretrained word embeddings](http://nlp.stanford.edu/data/glove.6B.zip)

### `classifier_rnn.ipynb`
In this file, a data preparation pipeline was developed to clean and normalize the dataset, and convert text from human language to machine-readable format for further processing.

The whole dataset is split into a training set and a test set to get an accurate measure of how well the model performs. To make sure the order of the data doesn’t influence the training process, randomization was performed during the process of training set and test set split.

In the model building part, pretrained word embeddings are used to build the embedding layer, Keras functional API is used to build the model, and Dropout is applied between layers to prevent overfitting.

### `predict.ipynb`
In this file, it calls a trained model from `classifier_rnn.ipynb` to make a prediction when a new consumer complaint narrative is entered.

## Future work
With enough computing power, a k-fold cross-validation can be used to get a more accurate measure of the model’s performance.

Here are some ideas that can further improve the model:
* Increase the size of the data
* Add more memory units to the layers and/or more layers.
* Increase the number of epochs

