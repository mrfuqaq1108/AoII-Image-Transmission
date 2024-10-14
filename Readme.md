# Related Paper
- Baojie Fu, Dapeng Wu and Ruyan Wang, Age of Incorrect Information in NOMA Aided
Image Transmission System

***Any use of the codes should explicitly cite the aforementioned paper.***

# Dataset
- Animals-10, which is available online at  https://www.kaggle.com/datasets/alessiocorrado99/animals10/data/

# Documentation
## Initialization (some simple file operations)
- **rename.py**: Rename all the files in the folder from 0000 to 9999.
- **delete_folder.py**: Delete all the files in the folder.
- **copy_file.py**: From folder '**raw-img/dataset**', copy the first 800 files to folder '**raw-img/train'**, copy the middle 300 files to folder '**raw-img/valid**' and the rest to folder '**raw-img/test**'.
- **build_new_SNR.py**: Obtain images with different probabilities of successful decoding at the BS side.

## Data Fitting
- main.py: Training the image recognition system and predict.
Approach: transfer learning, the pre-trained model is Resnet 50.

Dataset (for each category): 
- Training set: 0000~0799
- Validation set: 0800~1099
- Test set: 1100~end.

Select the sixth training parameters: '**raw-img_model_5.pt**'.

Prediction Accuracy: Number of successful identifications / Total Numbers.

- **figure_nn_loss.py**: Print figure (Loss&Accuracy versus Epoch Number)**(Figure 3)**
- **diff_SNR.py**: Calculate the image recognition accuracy under different probabilities of successful decoding.
- **SNRvsERROR.py**: Fit the original data by generalized logistic function and print both curves.**(Figure 4)**

## opt_AoII
- **new_algorithm.py**: Code for **Algorithm 1: PAoI-optimal Linear Search**, get the matrix **V**. Get **Fig.5(a)** and **Fig.5(b)**.
- **SINR_threshold.py**: Image Recognition Accuracy under different SINR threshold.Get **Fig.6(a)**.
- **power_acc.py**: Image Recognition Accuracy under different transmit power.Get **Fig.6(b)**

our proposed: Considering user fairness. Performance is the same curve for all users.

user 1 ~ user 6: Performance curves for user 1 to user 6.

average: The average performance curve for user 1 to user 6.

- **AoII_miu.py**: Impacts of different transmit power on AoII performance under different lambda and miu.
Get **Fig.7(a)** and **Fig.7(b)**.

