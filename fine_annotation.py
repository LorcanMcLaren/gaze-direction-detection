import pandas as pd
import numpy as np


video_name = '<video_file_name>' # Video file name without extension (e.g. mov)

gaze_df = pd.read_csv(video_name + "_df.csv")

# CONVERTING ORDINAL DIRECTIONS TO ANNOTATION LABELS - APPROPRIATE LABEL SHOULD BE UNCOMMENTED
# WHILE OTHERS REMAIN COMMENTED OUT
for index, row in gaze_df.iterrows():
    if row['Direction'] == 'Left':
        gaze_df.loc[index,'Direction'] = 'Gaze_Player-Left'
        # gaze_df.loc[index,'Direction'] = 'Gaze_Player'
        # gaze_df.loc[index,'Direction'] = 'Gaze_Away'
        # gaze_df.loc[index,'Direction'] = 'Gaze-Facilitator'
    elif row['Direction'] == 'Right':
        gaze_df.loc[index,'Direction'] = 'Gaze_Player-Right'
        # gaze_df.loc[index,'Direction'] = 'Gaze_Player'
        # gaze_df.loc[index,'Direction'] = 'Gaze-Facilitator'
    # elif row['Direction'] == 'Gaze_Away':
    #     gaze_df.loc[index,'Direction'] = 'Gaze_Player'

# CREATING DATAFRAME OF ANNOTATIONS WITH START TIME, END TIME AND LABEL 
# BASED ON FRAME-BY-FRAME ANALYSIS RESULTS
new_df = pd.DataFrame(columns=['Begin Time', 'End Time', 'Direction', 'Duration'])
prev_dir = gaze_df.loc[1, 'Direction']
prev_timestamp = 0.0
for index, row in gaze_df.iterrows():
    if prev_dir != row['Direction']:
        curr_timestamp = row['Timestamp']
        duration = curr_timestamp - prev_timestamp
        new_df = new_df.append({'Begin Time': prev_timestamp, 'End Time': curr_timestamp, 'Direction': prev_dir, 'Duration': duration}, ignore_index=True)
        prev_timestamp = curr_timestamp
    prev_dir = row['Direction']
    
curr_timestamp = gaze_df['Timestamp'].iloc[-1]
duration = curr_timestamp - prev_timestamp
new_df = new_df.append({'Begin Time': prev_timestamp, 'End Time': curr_timestamp, 'Direction': prev_dir,'Duration': duration}, ignore_index=True)


# REMOVING ENTRIES <= 100ms UNLESS BLINKING
new_df = new_df[((new_df['Duration'] > 100) | (new_df['Direction'] == 'Blinking'))]
new_df.index = range(len(new_df))

for index,row in new_df.iterrows():
    if index >= 1 and new_df.loc[index-1,'End Time'] != new_df.loc[index,'Begin Time']:
        new_df.loc[index,'Begin Time'] = new_df.loc[index-1,'End Time']
       

# MERGING NEIGHBOURING ROWS WITH SAME ANNOTATION VALUE    
prev_dir = 'none'
rows_to_drop = []
for index,value in new_df['Direction'].items(): 
    if value == prev_dir:
        new_df.loc[index,'Begin Time'] = new_df.loc[index-1,'Begin Time']
        new_df.loc[index,'Duration'] = new_df.loc[index,'End Time'] - new_df.loc[index,'Begin Time']
        rows_to_drop.append(index-1)
    prev_dir = value

new_df = new_df.drop(rows_to_drop)


# ROUNDING TO WHOLE TIME UNITS
new_df['Begin Time'] = np.floor(new_df['Begin Time'])
new_df['End Time'] = np.floor(new_df['End Time'])
new_df['Duration'] = np.floor(new_df['Duration'])


# CONVERTING TO INTS
new_df['Begin Time'] = pd.to_numeric(new_df['Begin Time'], downcast='signed')
new_df['End Time'] = pd.to_numeric(new_df['End Time'], downcast='signed')
new_df['Duration'] = pd.to_numeric(new_df['Duration'], downcast='signed')

# OUTPUTTING TO .CSV
annotation_name = video_name + "_annotation.csv"
new_df.to_csv(annotation_name, index=False)
