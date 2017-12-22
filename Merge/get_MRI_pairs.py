####################################################################
# Temporary script
####################################################################

import os
import pandas as pd
import numpy as np
from ast import literal_eval as make_tuple

# Uses GitHub filepaths
MRI_FILEPATH = 'MRI_Availability.csv'
SAVE_FILEPATH = 'MRI_Pairs.csv'
# Read the Files
df_mri = pd.read_csv( MRI_FILEPATH )

# Iterate through all the MRI data and match
mri_pair_column = df_mri.columns.tolist()[0]

PAIRS = []
for index, row in df_mri.iterrows():
	if 'nan' not in row[mri_pair_column]:
		element = make_tuple( row[mri_pair_column] )

		label = element[1]
		PAIRS.append( [element[0], label] )

df = pd.DataFrame( sorted(PAIRS), columns=['RID', 'VISCODE'] )
df.to_csv(SAVE_FILEPATH, index=False)