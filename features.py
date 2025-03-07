#%%

import pandas as pd
import numpy as np
import tensorflow as tf
from snowflake.snowpark import functions as F
from snowflake.snowpark.window import Window
from snowflake.snowpark.types import StructType, StructField, IntegerType, StringType

def extract_transition_features(session, df):
    
    depth_width_df = (df
     .with_column('depth',F.col('y') - F.col('spot_y'))
     .with_column('width',F.col('x') - F.col('spot_x'))
    )

    x_bins = [-50,-10,10,50]
    y_bins = [-100, -20, -10, 0, 15]

    x_labels = ["left third","middle","right third"]
    y_labels = ["real deep", "deep", "hooks", "blitz"]


    x_schema = StructType([
        StructField("xzone", StringType()),
        StructField("x_bin_min", IntegerType()),
        StructField("x_bin_max", IntegerType())
    ])

    y_schema = StructType([
        StructField("yzone", StringType()),
        StructField("y_bin_min", IntegerType()),
        StructField("y_bin_max", IntegerType())
    ])

    x_bin_data = [(x_labels[i], x_bins[i], x_bins[i + 1]) for i in range(len(x_labels))]
    y_bin_data = [(y_labels[i], y_bins[i], y_bins[i + 1]) for i in range(len(y_labels))]

    x_bin_df = session.create_dataframe(x_bin_data, schema = x_schema)
    y_bin_df = session.create_dataframe(y_bin_data, schema = y_schema)

    xzones = [
        "left third", "left third", "left third", "left third", 
        "middle", "middle", "middle", "middle", 
        "right third", "right third", "right third", "right third"
    ]

    yzones = [
        "real deep", "deep", "hooks", "blitz", 
        "real deep", "deep", "hooks", "blitz", 
        "real deep", "deep", "hooks", "blitz"
    ]

    zones = [
        "third_left","third_left","flat_left","blitz",
        "middle","middle","hooks","blitz",
        "third_right","third_right","flat_right","blitz"
    ]

    zone_grid = pd.DataFrame({
        'XZONE':xzones,
        'YZONE':yzones,
        'ZONE':zones
    })

    zone_df = session.create_dataframe(zone_grid)

    zone_order = ['blitz', 'flat_left', 'flat_right', 'hooks', 'middle', 'third_left', 'third_right']

    all_zone_possibilities = (df
     .select('clip_id','frameId')
     .drop_duplicates()
     .cross_join(zone_df.select('zone').drop_duplicates())
    )

    window_spec = Window.partition_by(['clip_id','zone']).order_by('frameId')

    transition_df = (depth_width_df
     .join(
         x_bin_df,
         (depth_width_df['width'] >= x_bin_df['x_bin_min']) & (depth_width_df['width'] < x_bin_df['x_bin_max']),
         how = "left"
        )
     .join(
         y_bin_df,
         (depth_width_df['depth'] >= y_bin_df['y_bin_min']) & (depth_width_df['depth'] < y_bin_df['y_bin_max']),
         how = 'left'
        )  
     .join(
         zone_df,
         ['xzone','yzone']
        ) 
     .filter(F.col('side') == 'defense')
     .groupBy(['clip_id','frameId','zone'])
     .agg(F.count("*").alias('players'))
     .join(
         all_zone_possibilities,
         ['clip_id','frameId','zone'],
         'outer'
        )
     .with_column('players', F.coalesce(F.col('players'),F.lit(0)))
     .with_column('players_prev', F.lag('players', 1).over(window_spec))
     .with_column('transition', F.col('players') - F.col('players_prev'))
    )

    counts_df = (transition_df
     .with_column('zone',F.upper(F.col('zone')))
     .group_by(['clip_id','frameId'])
     .pivot('zone', values = [x.upper() for x in zone_order])
     .agg(F.min('players'))
     .to_pandas()
    )

    changes_df = (transition_df
     .with_column('zone',F.upper(F.col('zone')))
     .group_by(['clip_id','frameId'])
     .pivot('zone', values = [x.upper() for x in zone_order])
     .agg(F.min('transition'))
     .to_pandas()
    )

    zone_counts = (counts_df
     .fillna(0)
     .merge(
         changes_df.fillna(0),
         on = ['CLIP_ID','FRAMEID'],
         how = 'left'
        )
     .sort_values(['CLIP_ID','FRAMEID'])
     .assign(min_frame = lambda x: x.groupby('CLIP_ID')['FRAMEID'].transform('min'))
     .query('FRAMEID != min_frame')
    )

    zone_counts_np = []

    for clip_id in zone_counts['CLIP_ID'].unique():

        zone_count_np = (zone_counts
         .query('CLIP_ID == @clip_id')
         .drop(['CLIP_ID','FRAMEID','min_frame'],axis = 1)
         .to_numpy()
        )

        zone_counts_np.append(zone_count_np)
    
    return zone_counts_np

def extract_player_features(df, side):

    model_features = ['X','Y','DIS','S','O','DIR']

    side_df = (df
     .filter(F.col('side') == side)
     .select(['clip_id','frameId','nflId'] + model_features)
     .to_pandas()
     .sort_values(['CLIP_ID','FRAMEID','NFLID'])
    )

    # clip_features = []

    # for clip_id in side_df['CLIP_ID'].unique():
    
    #     side_group = side_df.query('CLIP_ID == @clip_id').groupby(['FRAMEID'])

    #     frame_features = [np.concatenate(y[model_features].to_numpy()) for _, y in side_group]

    #     frame_features = np.stack(frame_features,axis = 0)    

    #     clip_features.append(frame_features)

    clip_groups = side_df.groupby('CLIP_ID')

    side_features = []

    for _, clip in clip_groups:

        frame_groups = clip.groupby('FRAMEID')

        frame_features = [np.concatenate(y[model_features].to_numpy()) for _, y in frame_groups]

        clip_features = np.stack(frame_features,axis = 0)    

        side_features.append(clip_features)

    return side_features

def process_static_features(df):

    static_features_= (df
     .select('clip_id','down','yardsToGo','spot_y')
     .drop_duplicates()
     .to_pandas()
     .sort_values('CLIP_ID')
     [['DOWN','YARDSTOGO','SPOT_Y']]
     .to_numpy()
    )    

    return static_features_

def process_play_data_w_mask(session, df, labels, mask_value):

    static_features = process_static_features(df)

    transition_features = extract_transition_features(session, df)
    defender_features = extract_player_features(df, side = 'defense')
    offense_features = extract_player_features(df, side = 'offense')
    
    # Standardize sequence length if needed
    max_transition_len = max(tf.shape(t)[0] for t in transition_features)
    max_defender_len = max(tf.shape(d)[0] for d in defender_features)
    max_offense_len = max(tf.shape(d)[0] for d in offense_features)
    
    # Pad sequences
    padded_transitions = tf.keras.preprocessing.sequence.pad_sequences(
        transition_features, 
        maxlen=max_transition_len,
        padding='post',
        dtype='float32',
        value = mask_value
    )
    
    padded_defenders = tf.keras.preprocessing.sequence.pad_sequences(
        defender_features,
        maxlen=max_defender_len,
        padding='post', 
        dtype='float32',
        value = mask_value
    )

    padded_offense = tf.keras.preprocessing.sequence.pad_sequences(
        offense_features,
        maxlen=max_offense_len,
        padding='post', 
        dtype='float32',
        value = mask_value
    )
    
    return padded_transitions, padded_defenders, padded_offense,  static_features, np.array(labels)