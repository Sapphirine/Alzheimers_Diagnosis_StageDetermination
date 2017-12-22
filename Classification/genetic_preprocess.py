####################################################################
# Preprocess the Genetic Data
####################################################################

import os
import pandas as pd
import numpy as np
import numpy.matlib
from numpy import linalg as LA
import math
import matplotlib.pyplot as plt

####################################################################
#Script Parameters
####################################################################

COL_LABELS = ['DX_bl']
COL_IDS = ['PTID']
COL_REMOVE = [ 'PTID.bl', 'VISCODE', 'COLPROT', 'ORIGPROT']

CLI_FILEPATH = '../Data/Genetic/adni_clin_gwas.csv'
SAVE_FILEPATH = 'GENETIC_FILTERED.csv'

####################################################################
#Step 1-1: Grab the relevant data
####################################################################

df_cli_data = pd.read_csv( CLI_FILEPATH )

INDEX_TRAIN = []
for index, row in df_cli_data.iterrows():
	if ( row['DX_bl'] == 'CN' 
			or row['DX_bl'] == 'AD' 
			or row['DX_bl'] == 'LMCI'
			or row['DX_bl'] == 'EMCI'):

		INDEX_TRAIN.append( index )

# Create new processed dataframe
df_cli_data = pd.DataFrame(df_cli_data.iloc[INDEX_TRAIN], columns=df_cli_data.columns)
df_cli_data = df_cli_data.reset_index(drop=True)

### Remove the non-feature columns except for DX_bl ##
df_features = df_cli_data.drop(COL_REMOVE, axis=1)

NUM_SAMPLES = len(df_features)
print( 'Number of Data Points: %d' % (NUM_SAMPLES) )

####################################################################
#Step 2-2: Record index of CN and AD patients
####################################################################
INDEX_CN = []
INDEX_AD = []
for index, row in df_features.iterrows():
	if row['DX_bl'] == 'CN':
		INDEX_CN.append( index )
	elif row['DX_bl'] == 'AD' or row['DX_bl'] == 'LMCI'or row['DX_bl'] == 'EMCI':
		INDEX_AD.append( index )

### Remove DX_bl ##
df_labels = df_features[ COL_LABELS ]
df_features = df_features.drop(COL_LABELS, axis=1)

### Remove RID, VISCODE ##
df_id = df_features[ COL_IDS ]
df_features = df_features.drop(COL_IDS, axis=1)

categories = []
for index, row in df_labels.iterrows():
	element = row['DX_bl']
	if element == 'EMCI' or element == 'LMCI':
		element = 'AD'
	if element not in categories:
		categories.append( element )
	df_labels.at[index, 'DX_bl'] = categories.index( element )

####################################################################
#Step 1-2: Record index of CN and AD patients
####################################################################

# Convert to a numpy array for PCA transformation
features = df_features.as_matrix()
features = features.astype(float)

### Each feature has roughly 0 mean, 1 var ###
feat_mean = np.mean(features, axis = 0)
feat_std = np.std(features, axis = 0)

features = np.subtract( features, 
	np.matlib.repmat(feat_mean, NUM_SAMPLES, 1) )
features = np.divide( features,
	np.matlib.repmat(feat_std, NUM_SAMPLES, 1) )

### Now that the features have similar statistics, perform PCA ###
row, col = np.shape(features)
cov = np.zeros([col, col])

for i in range(row):
	outer_prod = (1/row) * np.outer(features[i], features[i])
	cov = np.add(cov, outer_prod)
w, v = LA.eig( cov )

# For now don't do any dimensionality reduction

#Apply the Eigenvector transformation
features = np.matmul(features, v.real)

####################################################################
#Step 2: Visualize and Cluster/Classify
####################################################################
plt.scatter(features[INDEX_CN,0], features[INDEX_CN,2], alpha=0.5)
plt.scatter(features[INDEX_AD,0], features[INDEX_AD,2], alpha=0.5)
# plt.scatter(features[INDEX_LMCI,0], features[INDEX_LMCI,2], alpha=0.5)
# plt.scatter(features[INDEX_EMCI,0], features[INDEX_EMCI,2], alpha=0.5)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(['CN', 'AD'])
# plt.legend(['CN', 'AD', 'LMCI', 'EMCI'])
plt.show()

df_processed_feat = pd.DataFrame(features)
df_processed_feat = pd.concat([df_id, df_labels, df_processed_feat], axis=1)

# Save processed data
df_processed_feat.to_csv(SAVE_FILEPATH, sep=',', index=False)
print('Saved processed features as: %s' % (SAVE_FILEPATH))