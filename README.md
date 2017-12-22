# Alzheimers_Diagnosis_StageDetermination
EECS 6893 Big Data Analytics - Final Project

**Project ID**: 201712-18

**Authors**: Jing Ai (ja3130), Michael Nguyen (mn2769), Haoquan Zhao (hz2441)

Alzheimer’s Disease affect 1 in 3 seniors in the US and is one of the fastest rising part of the healthcare budget. Gaining better understanding of disease patterns and achieving accurate diagnosis showing the disease progression are crucial problems to address. Our project aims to identify the Alzheimer’s Disease biomarker combinations with the highest diagnostic power and examine the disease patterns of patients at different disease stages. The novelty of our project is that we performed a comprehensive analysis that integrated clinical, genomic and imaging data and included patients of multiple disease stages (Normal, Early Mild Cognitive Impairment, Late Mild Cognitive Impairment and Diagnosed Alzheimer’s), as previous studies have only focused their analysis on one modality and binary phenotypes (diseased/not diseased). 

## Dataset: ADNI data collection
The Alzheimer's Disease Neuroimaging Initiative (ADNI) data collection is a publicly available data collection consist of clinical, genetic and imaging datasets based on studies of approximately 1,550 participants including Alzheimer’s disease patients, mild cognitive impairment subjects and elderly controls across 3 multi-year cohorts (ADNI1, ADNI GO, ADNI2) between 2004 and 2017. More information on the data collection can be found here: http://adni.loni.usc.edu/data-samples/

## Analytics 

### Data Normalization, Preprocessing using PCA

#### Preprocessing
Clinical  	
- [*clinical_preprocess.py*](/Classification/clinical_preprocess.py)

Imaging - radiology measurements 
- [*imaging_preprocess.py*](/Classification/imaging_preprocess.py)

Genetic
- [*genetic_preprocess.py*](/Classification/genetic_preprocess.py)

Combined
- [*combined_preprocess.py*](/Classification/combined_preprocess.py)

#### Data-merging
- [*MergeMRI_radiology_measurements.ipynb*](/Merge/MergeMRI_radiology_measurements.ipynb)
- [*Merging_Clinical_Genetic_Imaging.ipynb*](/Merge/Merging_Clinical_Genetic_Imaging.ipynb)


### Classification, Features importance, Features correlations and Data visualization

Multi-layer Perceptron for Merged Data Classification
- [*mlp.py*](/Classification/mlp.py)

Convolutional Neural Network(CNN) for Raw Images Classification
- [*Final_project_visualization_raw_images.ipynb*](/CNN/Final_project_visualization_raw_images.ipynb)
- [*Transfer_learning.py*](/CNN/transfer_learning.py)
- [*Transfer_learning_adni.ipynb*](/CNN/transfer_learning_adni.ipynb)

Random Forest for classification and assessing feature importance
- [*adni_visualization.R*](/Rscript/adni_visualization.R)
- [*training_adni.R*](/Rscript/training_adni.R)

Spearman’s correlation for estimating feature correlations 
- [*adni_visualization.R*](/Rscript/adni_visualization.R)

tSNE for visualizing high dimensional data 
- [*tSNE.ipynb*](/Visualization/tSNE_on_merged_data.ipynb)
