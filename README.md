# CV_SeverstalSteelDefect
 Steel plate defect detections using computer vision

## Description
Kaggle dataset/competition: https://www.kaggle.com/c/severstal-steel-defect-detection/data

## Objective
Detect steel plate defects from images

## ML keywords:
* Computer Vision
* Semantic segmentation
* Dice Loss / Cross Entropy Loss
* Transfer learning
* FCN
* Pytorch

## Overview

12,000 256x1600 images of which about 50% have no defect and the other half has 1 or more defect classes.

![](/images/defect_image_initial.jpg)
![](/images/defect_image_ground_truth.jpg)
![](/images/defect_image_CE.jpg)
_Top: initial image  
Middle: image with ground truth mask  
Bottom: final model predictions_

## Approach

Semantic segmentation: assign a label 0 to 4 to each pixel, where 0 represents background and 1, 2, 3, 4 represent defect classes.  

Use FCN32 architecture with pretrained VGG11 for feature extraction.

## Accuracy metrics

Competition uses Dice function:  
For each label, create a mask of 0/1 values indicating if pixel belongs to that class.  
Dice = 2* | X ^ Y |  / (|X| + |Y|)
with X the set of pixels from ground truth and Y the prediction.  
Tensor implementation: 2* sum(X*Y)/(sum(X+Y)), where X and Y H x W tensors of 0/1 values
We'll refer to this function as Exact Dice.  

This function is not differentiable, so we'll transform X 0/1 values with "probability" values using softmax. We'll refer to this functions as smooth Dice.  

Initial Results: training the model with smooth dice loss  

![](/images/DiceError_DiceModel.jpg)

Model predictions

![](/images/defect_image_ground_truth.jpg)
![](/images/defect_image_Dice.jpg)
_Top: Actual defects
Bottom: predicted defects_

Results don't look so bad, so why such a high loss (around 71% on test set)? Also, evaluating the model on exact dice function, the accuracy is only 15%.

Predictions on an image without defects

![](/images/no_defect_image_initial.jpg)
![](/images/no_defect_image_Dice.jpg)
_Top: Actual defects
Bottom: predicted defects_

The model predicts isolated defect pixels. This makes the model unusable for production as it predicts many false positives.

![](/images/test_norm_Dice_confMat.jpg)
_Rows represent actual number of defect classes
Columns represent predicted number of defect classes
The model always predicts a few pixels from each class_

Final Results: training the model with Cross entropy

![](/images/CrossEntropyError_CEModel.jpg)

![](/images/defect_image_ground_truth.jpg)
![](/images/defect_image_Dice.jpg)
![](/images/defect_image_CE.jpg)
_Top: Actual defects  
Middle: Predictions using Dice training  
Bottom: Predictions using Cross entropy training_

Even though both models seem comparable at image level, the confusion matrix is a lot cleaner when training the model with cross entropy.

![](/images/test_norm_CE_confMat.jpg)

Also, on test set, exact dice accuracy jumps to 75% (from 15% using smooth dice for training).  On the validation set, accuracy is 72%.  



Next step: library https://github.com/qubvel/segmentation_models.pytorch for additional model testing with transfer learning.
