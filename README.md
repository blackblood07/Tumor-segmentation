# Tumor segmentaion with Brain MRI scans/ Kaggle

![brain tumor image](https://user-images.githubusercontent.com/112579358/235495039-a71d5602-c4d7-4081-aca2-c88d26d766a6.jpg)



  This repository holds an attempt to apply Convolutional Neural Networks (CNN) on Brain MRI scans using data from " Kaggle” (https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)

### Overview: 

  The task is to use the MRI- scans of actual cancer patients to perform active segmentation that could make treatments much faster. In modern technology, 
  Radiotherapy is the main treatment for cancer patients, the idea is to pass radiations on to the tumor to eradicate them. However,the surrounding areas shouldn't be affected and since 
  the segementation process is done manually by the doctors, it take hours for them to treat the patient. By developing a machine learning solution which automates the process of 
  tumor segmentation, the treatment can be administered in a faster rate which could save a lot of time and improve the entire procedure into more efficient one. 
  The approach in this repository formulates the problem as a semantic segmentation task, using convolutional neural networks as the model with pre-processed medical images as input. 
  Our best model was able to locate tumor closely with accuracy of “1.00” and with best loss value of “2.8355e-01”.

### Summary of work done:

### Data
  - The data used for this project was purely medical images of brain MRI scans, which is of type ".tif ". The size of each images is (256x256).
    The input taken is the original medical images and the respected output were the Segmented original medical images. The size of the entire datset is 749 MB. 
    In total the data set consisted of 7858 patients in which I limited the number of samples to 1500 images to be used for further processing and analysis. 
    60% of the sample (900 images) were used for training, 20% for validation (300 images) and 20% for testing (300 images).

### Preprocessing / Clean up

    - Seperated image paths and mask paths. 
    - Normalized pixel's brightness, anything above 127(Threshold) was consideres as white pixel(1) and rest were black(0).

### Data Visualization:

  Effective training visualisations were obtained after training:
          
  Loss Vs Epochs:
  
  ![image](https://user-images.githubusercontent.com/112579358/235492692-426127ff-d9bb-45f5-870e-c64b3ffce8df.png)


  F1 score Vs Epochs:
  
  ![image](https://user-images.githubusercontent.com/112579358/235493193-edd9b4db-c7d1-48ec-b582-434a4938a703.png)



    
### Problem Formulation

    - images and masks of brain MRI from kaggle data card were used as inputs.As an output segmented images with ground truth and prediction was obtained.

    * Model:
        " U- net " was used, as this model is known for working better with very few training samples while providing better performance for segmentation tasks.
    * Loss fucntions:
        Tried using loss functions such as BCEWithLogitsLoss,MSELoss,BCELoss,CrossEntropyLoss and DiceLoss 
        and was able to achieve loss values out of the loss fuction called "DiceLoss" 
    * Optimizer- Adam optimizer was used in this process. 

### Training

 The Software used for this project is "Kaggle"(Notebook environment) and the respective hardware used was my laptop. The training took about 51 minutes to complete,
 training curves (loss vs epoch for test/train and F1 vs epochs) are uploaded above. I decided to stop my training by checking where overfitting occured, meaning where the errors increased steeply and then decreased, the training curves came out pretty good and 100% accuracy was acheived at the 93rd epoch. The loss value was decreasing with good difference as well. During the process, I had a very difficult time training more that 100 images and my predictions were flipped. So I checked for the orders of the images and made sure the length of the images and the masks were off same size before training and towards the end increasing the image size to 1500 had me train them sucessfully. 
   - The key performance metrics used was F1 score


### Results:

Image, Ground truth, prediction:

![image](https://user-images.githubusercontent.com/112579358/235493478-414e1e26-eaf4-496c-943d-743f5fe46cf7.png)


![image](https://user-images.githubusercontent.com/112579358/235493572-67a27e60-2722-44e0-823c-1df73627f08b.png)


### Conclusions

   - In conclusion,The tumor segemnatation was actively done by obtating the prediction closer to the ground truth segementation even with fewer images.
      So, U-net works better in the case of reduced number of images 

### Future Work

   - In the future, I would try to train more images using double U-net as it is more refined and results of accuracy can be even more close and perfect. So, I will work to check if my assumptions 
     make sense or not. Concept of segementation can be studied and applied in various medical field beyond brain tumor segementation, such as GI tract tumor, Lung cancer, Breast cancer and so on. 
     Additionally, the concept of segementataion can be apllied outside of medical field such as obejct detection in autonomous vehicles and Robotics.
 


### How to reproduce results

   - To reproduce the results, the user must have Python 3.6 or higher installed, as well as the PyTorch and segmentation-models-pytorch packages. This can be effectively done in Kaggle environment too as I did. 
     These can be installed via pip by running the commands as per "Necessary dependencies.py" file uploaded above. Data preparation can be done in your own ways as the dataset has no limitations. 
     Basic preprocessing step should be done. Make sure to normalize the pixels before any further analysis. To train a model on your own data, you can use the code in the "build_U-net.py file". 
     You will need to modify the Dataset class to read in their own data. Evaluation and prediction methods are also flexible you can refer the notebook if you prefer to evaluate based on F1 score and Loss values. 
     One thing to note is to make sure to select the GPU 4 option which can be found under the accerelator options if you were to use Kaggle notebook to significantly speed up the training process. 

### Overview of files in repository
  
  - To install the required packages please refer to "Necessary dependencies.py" file. Copy and paste the libraries listed and a simple cell run should have them installed in your environment. 
    If you hit up with some errors while installing, please make sure to pip install the packages before importing. Follow the link attached to the pip install guide.
    
    link: https://pip.pypa.io/en/stable/cli/pip_install/

### Data
  
  - If you are interested to use the same data with different approach based on this project, you can download the dataset here: https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation

### Training

  - initialize and prepare loaders using "data_prep.py" file. Build and define the model using "build_U-net.py" file. Train the model using "train_model.py"

### Citations

* https://github.com/zhixuhao/unet (U-net model original creator referenced)

* https://www.kaggle.com/code/samuelcortinhas/case-study-u-net-from-scratch#Application:-Tumor-detection

* https://www.kaggle.com/code/tejasurya/unet-from-scratch-segmentation-tumour/notebook

* https://www.kaggle.com/code/heiswicked/pytorch-unet-segmentation-tumour

* https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6975331/#:~:text=SUMMARY%3A,associated%20with%20various%20intracranial%20pathologies.
  




