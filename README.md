# Text classification

## Business Problem
The bank receives thousands of consumer complaints per week. The most important part of providing good customer service is properly managing consumer complaints. So How can we respond quickly and effectively to all these customers? 

## Solution
Customer complaint classifier with machine learning is the answer. It automatically routes the complaints to the right team, and analyze them for immediately actionable insights.

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

