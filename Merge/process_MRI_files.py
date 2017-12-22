####################################################################
# Process MRI files
#
# This script cycles through all csv files in the MRI image folder
#
# The output is a table where each row corresponds to a (RID, VISCODE) pair and
# the column indicates if that pair has data for a given file (0,1)
# The viscode labels have not been cleaned, ADNI1,2,GO data is represented in
# the table
####################################################################

import os
import pandas as pd
import numpy as np

# Uses personal filepaths
FILEPATH = '../ADNI_MRI_Analysis/'
SAVEPATH = 'MRI_Availability.csv'
files = os.listdir( FILEPATH )

## Obtain the relevant files for processing
# Get only the csv files
files = [file for file in files if file.endswith('.csv')]

# Removes the DICT files
files = [file for file in files if 'DICT' not in file]

# Removes the ADNI1 files (if explicitly defined in the file title)
files = [file for file in files if 'ADNI1' not in file]

# Number of files we will be potentially using for features
NUM_FEATS = len( files )
FEAT_FILE = []
for file in files:
	FEAT_FILE.append( file )

FEAT_PAIRS = {}
count = 0

for file in files:
	print( 'Reading file (%i/%i)'  % (count + 1, NUM_FEATS) + ': ' + file + '...' )

	df = pd.read_csv( FILEPATH + file )

	# Tries to use VISCODE2 --> VISCODE1 --> VISCODE (in order of priority)
	if 'VISCODE2' in df.keys():
		viscode_str = 'VISCODE2'
	elif 'VISCODE1' in df.keys():
		viscode_str = 'VISCODE1'
	else:
		viscode_str = 'VISCODE'

	# For each entry in the table:
	for index in range( len( df ) ):
		row = df.iloc[index]

		# Establish the key

		rid = row['RID']
		viscode = row[viscode_str]
		key = (rid, viscode)

		# Update the master table
		if key not in FEAT_PAIRS.keys():
			FEAT_PAIRS[key] = np.zeros( NUM_FEATS )
		FEAT_PAIRS[key][count] = 1

	count += 1
output = pd.DataFrame(FEAT_PAIRS, index=FEAT_FILE)
output = output.transpose()
output.to_csv(SAVEPATH, sep=',')

print('[Done]')