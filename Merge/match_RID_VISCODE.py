####################################################################
# Match (RID, VISCODE)
#
# This script cycles through all CSV files contining RID, VISCODE 
# pairs.
#
# The matched pairs are 
####################################################################

import os
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# Uses GitHub filepaths
CLI_FILEPATH = 'Clinical_ADNI_GO_2_RID_VISCODE.csv'
MRI_FILEPATH = 'MRI_Pairs.csv'

SAVE_FILEPATH = 'matched_pairs.csv'

# Read the Files
df_cli = pd.read_csv( CLI_FILEPATH )
df_mri = pd.read_csv( MRI_FILEPATH )

# Obtain the (RID, VISCODE) tuple for MRI
RID_VISCODE_MRI_LIST = []
for index, row in df_mri.iterrows():
	element = (row['RID'], row['VISCODE'])
	RID_VISCODE_MRI_LIST.append( element )

# Obtain the (RID, VISCODE) tuple for Clinical
RID_VISCODE_CLI_LIST = []
for index, row in df_cli.iterrows():
	element = (row['RID'], row['VISCODE'])
	RID_VISCODE_CLI_LIST.append( element )

# Iterate through all the MRI data and match
MATCH_PAIRS = []
for element in RID_VISCODE_MRI_LIST:
	if element in RID_VISCODE_CLI_LIST:
		MATCH_PAIRS.append( element )

# Display number of matching pairs
print( 'Number of matching pairs: %i' % ( len(MATCH_PAIRS) ) )

df = pd.DataFrame( sorted(MATCH_PAIRS), columns=['RID', 'VISCODE'] )
df.to_csv(SAVE_FILEPATH, index=False)

# Get unique patients:
unique_patients = []
for element in MATCH_PAIRS:
	if element[0] not in unique_patients:
		unique_patients.append( element[0] )
unique_patients = sorted( unique_patients )

print( 'Number of unique patients: %i' % ( len(unique_patients) ) )

# Get frequency of viscode:
viscode = []
for element in MATCH_PAIRS:
	viscode.append( element[1] )
viscode = sorted( viscode )

viscode_counts = Counter( viscode )
df = pd.DataFrame.from_dict(viscode_counts, orient='index')
df.plot(kind='bar')
plt.show()