#%%

import keyring
from snowflake.snowpark.session import Session
from snowflake.snowpark import functions as F
from snowflake.snowpark.window import Window
import pandas as pd
import numpy as np

#%%

getin = keyring.get_password('snowflake','sgwoods')

connection_parameters = {
    'account':'HFRQIAT-KJB27936',
    'user':'sgwoods',
    'password':getin,
    'role':'SYSADMIN',
    'warehouse':'COMPUTE_WH',
    'database':'NFL',
    'schema':'tracking'
}

session = Session.builder.configs(connection_parameters).create()

#%%

tracking = session.table('tracking')

plays = session.table('plays')

#%%

tracking_w_side = (tracking
 .join(
    plays.select('gameId','playId','defensiveTeam'),
    ['gameId','playId']
   )
 .withColumn(
     'side',
      F.when(F.col('team') == "football","football")
      .when(F.col('team') == F.col('defensiveTeam'),"defense")
      .otherwise("offense")
   )
 .withColumn('o',F.col('o').cast('double'))
)

rename_cols = [x.upper() for x in ['nflId','x','y','s','a','o']]

prefix = 'o_'

renamed_columns = [(F.col(c).alias(f"{prefix}{c}") if c in rename_cols else F.col(c)) for c in tracking_w_side.columns]

rel_df = (tracking_w_side
 .filter(F.col('side') == 'defense')
 .select(['gameId','playId','frameId'] + rename_cols)
 .join(
    (tracking_w_side
     .filter(F.col('side').isin(["football","offense"]))
     .select(*renamed_columns)
     .withColumn('football',F.when(F.col('side') == "football",1).otherwise(0))
     .select(['gameId','playId','frameId','football'] + [prefix + x for x in rename_cols])
    ),
    ['gameId','playId','frameId'],
    'left'
   )
 .withColumn('diff_x',F.col('o_x') - F.col('x'))
 .withColumn('diff_y',F.col('o_y') - F.col('y'))
 .withColumn('diff_s',F.col('o_s') - F.col('s'))
 .withColumn('diff_a',F.col('o_a') - F.col('a'))
 )

model_features = [x for x in rel_df.columns if len(x) == 1] \
   + [x for x in rel_df.columns if 'DIFF' in x] \
   + ['FOOTBALL']

#%%

coverage_spec = Window.order_by(F.col('pff_passCoverage'))

rdws_w_coverage = (plays
 .with_column(
     'pff_passCoverage',
     F.when(F.col('pff_passCoverage') == '2-Man','Cover-2')
     .otherwise(F.col('pff_passCoverage'))
    )
 .with_column('coverage',F.dense_rank().over(coverage_spec))
)

coverage_factors = rdws_w_coverage.select('pff_passCoverage','coverage').dropDuplicates()

#%%

frames_by_play = (rel_df
 .groupBy('gameId','playId')
 .agg(F.countDistinct('frameId').alias('frames'))
 )

max_frames = 51

#%%

frame_padding = frames_by_play.withColumn('padding',max_frames - F.col('frames')).toPandas()

padded_frames = []

for frame_padding_row in frame_padding[['GAMEID','PLAYID','PADDING']].to_numpy():

    game_id, play_id, padding = frame_padding_row

    zeros_df = pd.DataFrame(0, index = np.arange(padding),columns = [x.upper() for x in model_features])

    return_df = zeros_df.assign(
        FRAMEID = range(max_frames - padding + 1,max_frames + 1,1),
        GAMEID = game_id,
        PLAYID = play_id
    )

    padded_frames.append(return_df)

padded_frames = pd.concat(padded_frames).reset_index(drop = True)

#%%

padded_frames_spk = session.write_pandas(
    padded_frames,
    'padded_frames',
    auto_create_table = True,
    table_type = 'temp',
    overwrite = True
   )

rows_to_add = (padded_frames_spk
 .join(
     rel_df.select('gameId','playId','nflId','o_nflId').drop_duplicates(),
     ['gameId','playId']
    )
)

clip_spec = Window.order_by(F.col('gameId'),F.col('playId'))

plays_w_clip_id = (plays
 .with_column('clip_id',F.dense_rank().over(clip_spec))
 [['gameId','playId','clip_id']]
)

#%%

min_frames = (rel_df
 .groupBy(F.col('gameId'),F.col('playId'))
 .agg(F.min(F.col('frameId')).alias('min_frame'))
 )

rel_df_adj_frames = (rel_df
 .join(
     min_frames,
     ['gameId','playId']
    )
 .with_column('frameId',F.col('frameId') - F.col('min_frame') + 1)
 )

#%%

rel_df_avec_padding = (rel_df_adj_frames
 [rows_to_add.columns]
 .union(rows_to_add)
 .join(
     plays_w_clip_id,
     ['gameId','playId']
    )
 .orderBy(['gameId','playId','frameId','nflid','o_nflid'])
)

rel_df_sans_padding = (rel_df_adj_frames
 .join(
     plays_w_clip_id,
     ['gameId','playId']
    )
 .orderBy(['gameId','playId','frameId','nflid','o_nflid'])
)

#%%

coverages_to_model = ['Cover-1', 'Cover-3', 
                      'Cover-6', 'Quarters', 
                      'Cover-2', 'Cover-0']

model_df = (rel_df_sans_padding
 .join(
     rdws_w_coverage.select('gameId','playId','pff_passCoverage'),
     ['gameId','playId']
    )
 .filter(F.col('pff_passCoverage').in_(coverages_to_model))
 .orderBy(['clip_id','frameId','nflid','o_nflid'])
 .select(['clip_id','pff_passCoverage'] + model_features)
 ## Had to manually remove this one because of error reasons
 .filter((F.col('gameId') != 2022092508) | (F.col('playId') != 3104))
)

model_df.write.save_as_table("model_df", mode = "overwrite", table_type = 'temp')

model_df = session.table("model_df")

#%%

coverage_mapping = (plays
 .filter(F.col('pff_passCoverage').in_(coverages_to_model))
 .select('pff_passCoverage')
 .drop_duplicates()
 .to_pandas()
 .assign(coverage = lambda x: pd.factorize(x['PFF_PASSCOVERAGE'])[0])
)

clip_ids = (model_df
 .select('clip_id')
 .drop_duplicates()
 .to_pandas()
)

from sklearn.model_selection import train_test_split
from math import ceil

train_clips, test_clips = train_test_split(clip_ids, test_size = 0.2, random_state = 42)
ana_clips, assess_clips = train_test_split(train_clips,test_size = .25, random_state = 15)

ana_clips, assess_clips, test_clips = [(df
 .reset_index(drop = True)
 .reset_index()
 .assign(bin = lambda x: pd.cut(x['index'],ceil(len(df)/1000)),
         chunk_id = lambda x: pd.factorize(x['bin'])[0])
 [['CLIP_ID','chunk_id']]
 .values
) for df in [ana_clips, assess_clips, test_clips]]

ana_chunk_ids = list(set([x[1] for x in ana_clips]))
assess_chunk_ids = list(set([x[1] for x in assess_clips]))

class_count_df = (model_df
 .filter(F.col('clip_id').in_([int(x[0]) for x in ana_clips]))
 .select('clip_id','PFF_PASSCOVERAGE')
 .drop_duplicates()
 .to_pandas()
 .merge(
     coverage_mapping,
     on = 'PFF_PASSCOVERAGE'
    )
 .groupby('coverage')
 .agg(count = ('CLIP_ID','size'))
 .reset_index()
)

coverages_to_model_ordered = class_count_df['coverage'].values
class_counts = class_count_df['count'].values