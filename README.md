# Binary image classification

 Identify the presence of metastases from 96 x 96px digital images. It includes data exploration, data preprocessing with a pipeline and classification model by using the fine-tuning of a model from **fast.ai**

## Evaluation metric

The evaluation metric is area under the ROC curve. The ROC curve is a plot of True positive rate against False positive rate at various thresholds and the area under the curve (AUC) is equal to the probability that a classifier will rank a randomly chosen positive instance higher than a randomly chosen negative one. The best possible solution would yield an AUC of 1 which means we would classify all positive samples correctly without getting any false positives.

