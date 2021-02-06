


#### EDA 
# Integrating and Exploring the Combined MSI Dataset for California

#### Data Processing 
1. BigEarthNet dataset (Surya to add more details)
2. BigEarthData Into TFRecords
	1. `src/data/preprocess_tfrecords.py` 
	2. `src/data/preprocess_tfrecords_labeled.py`  (irrigation vs non-irrigation)
	3. To preprocess the GeoTiff files that contain the Sentinel-2 MSI data comprising the BigEarthNet dataset into TFRecords files. It is based on the preprocessing scripts from the BigEarthNet repo, but has been updated to work in Colaboratory with Python3.7+ and TensorFlow 2.3.
	4. input: BigEarthNet dataset
		1.  v1: s3://BigEarthNet-v1.0
		2. bigearthnet-models/label_indices.json 



# Data Augmentation Pipeline Testing



#### Model Valuation 
 processing results from supervised and unsupervised models. We do this by plotting training and validation metrics by epoch for the supervised and finetuned models. We also evaluate models against our test set and look at AUC. What we are after is identifying what parameters during unsupervised training and finetuning lead to the highest overall AUC as compared to our supervised baseline which is identified in the cells below.
- `notebooks/BigEarthData/epoch_metrics notebook`
- 