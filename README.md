Project read me: Note- This is a work in progress!

# Tumor segmentaion with MRI scans/ Kaggle

### one sentence summary:

This repository holds an attempt to apply Convolutional Neural Networks (CNN) on Brain MRI scans using data from " Kaggle” to automate the process of segementation of tumor cells. (https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)

### Abstract:

  The task, is to use the MRI- scans of actual cancer patients who had taken MRI scans during their radiation treatment and peroform active segmentation that could   make treatment much faster. The approach in this repository formulates the problem as a semantic segmentation task, using convolutional neural networks as the model with pre-processed medical images as input. Our best model was able to locate tumor as closely as possible with accuracy of “ 0.97” and with best loss value of “ 9.4227e-01”. 

### Summary of work done:

### Data
    Type: medical images (256x256 pixel .tif),
          Input: original medical images, output: Segmented original medical images
    Size: 749 MB
   
    Instances:  (Train, Test, Validation Split): total 7858 patients,(decided to train, test and validate with 100 images), 60 for training, 20 for validation (60-80 images), 20 for testing (80- 100 images)

### Preprocessing / Clean up
    * Seperated image paths and mask paths. 

    * Normalized pixel's brightness, anything above 127 was consideres as white pixel(1) and rest were black(0).

### Data Visualization

Training visulaizations will be provided soon
    
### Problem Formulation

    Define:
    Input:- images and masks of brain MRI from kaggle data card were used as inputs.
    Output:- segmented images with ground truth and prediction was obtained.
   
    Models:
    * model " U- net " was used, as this model is known for working better with very few training samples while providing better performance for segmentation tasks.
    * Tried using loss functions such as BCEWithLogitsLoss,MSELoss,BCELoss,CrossEntropyLoss and DiceLoss 
      and was able to achieve loss values out of the loss fuction called "DiceLoss" 
    * Optimizer- Adam optimizer. 

### Training

#### Describe Training**:

How you trained: Software- Kaggle and hardware- Laptop.

How long did training take- less than 1 minute

 Training curves (loss vs epoch for test/train).
 
 How did you decide to stop training.
 
 - I checked where overfitting occured meaning where the errors increased steeply and then decreased.
 
 Any difficulties? How did you resolve them? 
- Right now my prediction is flipped and Im working on identifying the error, will get over that soon. 

### Performance Comparison
   - The key performance metrics used was F1 score
   
   Will be updating soon
   - Show/compare results in one table.
   - Show one (or few) visualization(s) of results, for example ROC curves

### Conclusions

-State any conclusions you can infer from your work: 

### Future Work
-What would be the next thing that you would try.

 -- I am trying to improve the result by training minimun of 500 images and maximun of 1000 images.
 
 -- The current machine learning framework being utilized is "pytorch," however, due to memory errors arising from this platform, I am attempting to modify and explore potential solutions to resolve the issue, and also experimenting with "tensorflow" to determine if it can address the problem at hand.
 
 -- My next implementation step will involve increasing the number of epochs to 500 and generating a Loss vs Epoch curve, which will enable me to draw better conclusions based on the graph and assess the effectiveness of the training process.
 
-What are some other studies that can be done starting from here.

### How to reproduce results

In this section, provide instructions at least one of the following:

  -Reproduce your results fully, including training.

  -Apply this package to other data. For example, how to use the model you trained.

  -Use this package to perform their own study.

  -Also describe what resources to use for this package, if appropirate. For example, point them to Collab and TPUs.

### Overview of files in repository

 -Describe the directory structure, if any.

### List all relavent files and describe their role in the package.

An example:

utils.py: various functions that are used in cleaning and visualizing data.
preprocess.ipynb: Takes input data in CSV and writes out data frame after cleanup.
visualization.ipynb: Creates various visualizations of the data.
models.py: Contains functions that build the various models.
training-model-1.ipynb: Trains the first model and saves model during training.
training-model-2.ipynb: Trains the second model and saves model during training.
training-model-3.ipynb: Trains the third model and saves model during training.
performance.ipynb: loads multiple trained models and compares results.
inference.ipynb: loads a trained model and applies it to test data to create kaggle submission.
Note that all of these notebooks should contain enough text for someone to understand what is happening.

### Software Setup

  List all of the required packages.
  If not standard, provide or point to instruction for installing the packages.
  Describe how to install your package.

 ### Data
 Point to where they can download the data.
 Lead them through preprocessing steps, if necessary.

### Training
  Describe how to train the model
  Performance Evaluation
  Describe how to run the performance evaluation.

### Citations

* https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation

* https://github.com/zhixuhao/unet

* https://www.kaggle.com/code/samuelcortinhas/case-study-u-net-from-scratch#Application:-Tumor-detection

* https://www.kaggle.com/code/tejasurya/unet-from-scratch-segmentation-tumour

* https://www.kaggle.com/code/bonhart/brain-mri-data-visualization-unet-fpn

* https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6975331/#:~:text=SUMMARY%3A,associated%20with%20various%20intracranial%20pathologies.
  




