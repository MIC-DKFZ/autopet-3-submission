# Welcome to our solution for the autoPET III challenge

This repository provides the code for running our approach to the [AutoPET III Challenge](https://autopet-iii.grand-challenge.org/). When using this repository, please cite our paper:

From FDG to PSMA: A Hitchhiker's Guide to Multitracer, Multicenter Lesion Segmentation in PET/CT Imaging 

&nbsp; &nbsp;   [![arXiv](https://img.shields.io/badge/arXiv-2404.03010-B31B1B.svg)](https://arxiv.org/abs/2404.03010) 

Authors:  
Maximilian Rokuss, Balint Kovacs, Yannick Kirchhoff, Shuhan Xiao, Constantin Ulrich, Klaus H. Maier-Hein and Fabian Isensee


Author Affiliations:  
Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg  
Faculty of Mathematics and Computer Science, Heidelberg University

# Overview

Our model builds on [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) with a [ResEncL architecture](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/resenc_presets.md) preset as a strong baseline. We then introduce several improvements:

- We adjusted the plans file using a patch size of  ```[192,192,192]``` suhc that the model has a high contextual understanding. You can find the the plans [here](nnunetv2/architecture/nnUNetResEncUNetLPlansMultiTalent.json) (remember to change the "dataset_name" field to your dataset name).
- The model is pretrained on a diverse collection of medical imaging datasets to establish a strong anatomical understanding, which we then fine-tune on the autoPET III challange dataset. The pretrained checkpoint is availabe [here](https://zenodo.org/records/13753413) (Dataset619_nativemultistem).
- The model is trained using [misalignment data augmentation](https://github.com/MIC-DKFZ/misalignment_DA) as well as omitting the smoothing term in the dice loss calcuation.
- We use a dual-headed architecture for organ and lesion segmentation which improves performance as well as speeds up convergence, especially in cases without lesions.

## Getting started

### Installation

We recommend to create a new conda environment and then run:


          ```bash
          git clone https://github.com/MIC-DKFZ/autopet-3-submission.git
          cd autopet-3-submission
          pip install -e .
          ```

### Preprocessing

Download the [autoPET dataset](https://autopet-iii.grand-challenge.org/dataset/) and use the standard nnUNet preprocessing pipeline. You can adjust the number of processes for faster processing. You can freely chose your dataset number which we quote as DATASET_ID_LESIONS.

          ```bash
        nnUNetv2_plan_and_preprocess -d DATASET_ID_LESIONS -c 3d_fullres -np 20 -npfp 20
          ```

### Extract organs from CT image

In order to train on the organ segmentation as a secondary objective we use [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) to predict 10 anatomical structures in part relevant to PET tracer uptake: spleen, kidneys, liver, urinary bladder, lungs, brain, heart, stomach, prostate, and glands in the head region (parotid glands, submandibular glands).

You can either:

1.    Use our predicted organ masks which we made availabe [in this repo](nnunetv2/preprocessing/organ_extraction/autopet3_organ_labels) 

2.    Follow these instructions for your own dataset: This one is a bit time-consuming to redo, so hang with me here. First, install TotalSegmentator using ```pip install TotalSegmentator```. We did it in a separate environment. Then put [this script](nnunetv2/preprocessing/organ_extraction/predict_and_extract_organs.py) into your nnUNet_raw dataset directory for autoPETIII. Running this script can take a long time for the whole dataset (several days on our machines) since the TotalSegmentator inference is not optimized to handle several cases simultaneously. 

          ```bash
          python predict_and_extract_organs.py
          ```

When this step is done, copy the raw nnUNet dataset such that you have a new dataset which is identical. The original dataset containing lesions annotations should have a separate DATASET_ID_LESIONS than the new DATASET_ID_ORGANS. E.g. "Dataset200_autoPET3_lesions" and "Dataset200_autoPET3_organs". Then exchange the content of the  ```labelsTr``` folder with the provided [organ labels](nnunetv2/preprocessing/organ_extraction/autopet3_organ_labels) or in case you ran the above script (TotalSegmentator inference) use the labels from ```labelsTr_organs```. Now run the preprocessing again for the new dataset. Important: do not use the ```--verify_dataset_integrity``` flag.

Lastly, to combine the datasets run

          ```bash
          nnUNetv2_merge_lesion_and_organ_dataset -l DATASET_ID_LESIONS -o DATASET_ID_ORGANS
          ```

Now you are good to go to start a training. Use the dataset with DATASET_ID_LESIONS for any further steps. If everything runs smoothly you could discard the dataset folder with DATASET_ID_ORGANS.

> If you want to do the last step step manually head over to the preprocessed folder. You should have two folders now with the different preprocessed datasets, containing lesion or organ segmentations. Navigate to the dataset containing the organ labels and then into the folder ```nnUNetPlans_3d_fullres```. Either unpack the ```.npz``` files using a script or start a default nnUNet training on the organ dataset such that they are automatically unpacked to ```.npy``` files. Now we are almost there. Search for all files ending with ```_seg.npy``` and rename them to have the ending ```_seg_org.npy```. Finally copy these files into the ```nnUNetPlans_3d_fullres``` folder of the preprocessed dataset containing the lesion segmentations. That's it - easy right?


### Training

Training the model can be simply achieved by [downloading the pretrained checkpoint](https://zenodo.org/records/13753413) (Dataset619_nativemultistem) and running:

          ```bash
          nnUNetv2_train DATASET_ID_LESIONS 3d_fullres 0 -tr autoPET3_Trainer -p nnUNetResEncUNetLPlansMultiTalent -pretrained_weights /path/to/pretrained/weights/fold_all/checkpoint_final.pth
          ```

We train a five fold cross-validation for our final submission.


Happy coding! 🚀

# Acknowledgements
<img src="documentation/assets/HI_Logo.png" height="100px" />

<img src="documentation/assets/dkfz_logo.png" height="100px" />

nnU-Net is developed and maintained by the Applied Computer Vision Lab (ACVL) of [Helmholtz Imaging](http://helmholtz-imaging.de) 
and the [Division of Medical Image Computing](https://www.dkfz.de/en/mic/index.php) at the 
[German Cancer Research Center (DKFZ)](https://www.dkfz.de/en/index.html).