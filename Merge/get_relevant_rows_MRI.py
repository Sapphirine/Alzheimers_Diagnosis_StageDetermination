####################################################################
# Insert description here
####################################################################

import os
import pandas as pd
import numpy as np

PAIR_FILEPATH = 'matched_pairs.csv'
MRI_FILEPATH = 'MRI_Availability.csv'

SAVE_FILEPATH = 'MRI_Availability_filtered.csv'

# Read the Files
df_pair = pd.read_csv( PAIR_FILEPATH )
df_mri = pd.read_csv( MRI_FILEPATH, index_col=0 )

# Obtain the (RID, VISCODE) str for Clinical
RID_VISCODE_LIST = []
for index, row in df_pair.iterrows():
	element = (row['RID'], row['VISCODE'])
	RID_VISCODE_LIST.append( str( element ) )

# Get the row labels (should be strings)
index_list = df_mri.index.tolist()

index_train = []
for element in RID_VISCODE_LIST:
	index = -1
	for ind, label in enumerate( index_list ):
		if element == label:
			index = ind
			break
	if index == -1:
		print("[ERROR]: Index not found")
		print( element )
		break
	index_train.append( index )



output = pd.DataFrame(df_mri.iloc[index_train], columns=df_mri.columns)
output.to_csv(SAVE_FILEPATH, sep=',', index_label=False)