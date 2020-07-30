import pandas as pd
import sys
from pathlib import Path

# filename = sys.argv[1]
# annotations = pd.read_csv(filename)


filename = "<fine_annotation_file>.csv"

annotations = pd.read_csv(filename)

# RESETTING INDICES FOLLOWING DELETIONS IN fine_annotation.py
annotations = annotations.loc[annotations['Direction'] != 'Blinking']
annotations.index = range(len(annotations))

# FILLING ANY GAPS CAUSED BY DELETIONS TO ENSURE UNBROKEN ANNOTATION
for index,row in annotations.iterrows():
    if index >= 1 and annotations.loc[index-1,'End Time'] != annotations.loc[index,'Begin Time']:
        annotations.loc[index,'Begin Time'] = annotations.loc[index-1,'End Time']

# MERGING 
prev_dir = 'none'
rows_to_drop = []
for index,value in annotations['Direction'].items(): 
    if value == prev_dir:
        annotations.loc[index,'Begin Time'] = annotations.loc[index-1,'Begin Time']
        annotations.loc[index,'Duration'] = annotations.loc[index,'End Time'] - annotations.loc[index,'Begin Time']
        rows_to_drop.append(index-1)
    prev_dir = value

annotations = annotations.drop(rows_to_drop)

# OUTPUTTING TO .CSV
name = Path(filename).stem + "_broad.csv"
annotations.to_csv(name, index=False)