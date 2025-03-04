#%%

import keyring
from snowflake.snowpark.session import Session
from snowflake.snowpark import functions as F
from snowflake.snowpark.window import Window
import pandas as pd
import numpy as np

#%%

login = keyring.get_password('snowflake','account')
getin = keyring.get_password('snowflake','sgwoods')

connection_parameters = {
    'account':login,
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

players = session.table('players')

#%%

events_to_clip = [
    'pass_forward','autoevent_passforward','qb_sack','run','pass_outcome_incomplete','pass_arrived',
    'autoevent_passinterrupted','qb_strip_sack','pass_tipped','fumble','pass_outcome_caught','handoff',
    'pass_shovel','qb_spike','pass_outcome_interception', 'lateral', 'tackle',
    'fumble_defense_recovered', 'fumble_offense_recovered', 'qb_slide',
    'pass_outcome_touchdown', 'dropped_pass', 'safety'
]

window = Window.partition_by('gameId','playId').order_by('frameId')

max_frame = (tracking
 .filter(F.col('event').in_(events_to_clip))
 .order_by('gameId','playId','frameId')
 .drop_duplicates(['gameId','playId'])
 .select('gameId','playId','frameId',F.first_value(F.col('event')).over(window).as_('first_event'))
)

tracking_clipped = (tracking
 .join(
     max_frame.select('gameId','playId',F.col('frameId').alias('max_frame')),
     ['gameId','playId'],
     'left'
    )
 .filter((F.col('frameId') <= F.col('max_frame')) | (F.col('max_frame').isNull()))
 )


#%%

columns_to_use = ['gameId','playId','nflId','frameId','team','playDirection','season',
                  'x','y','s','a','dis','o','dir']

tracking_w_side = (tracking_clipped
 .select(columns_to_use)
 .join(
     plays.select('gameId','playId','defensiveTeam','yardsToGo','absoluteYardLineNumber','down'),
     ['gameId','playId']
   )
.join(
     players.select('nflId','season','position'),
     ['nflId','season'],
     'left'
    )
 .withColumn(
     'side',
      F.when(F.col('team') == "football","football")
      .when(F.col('team') == F.col('defensiveTeam'),"defense")
      .otherwise("offense")
 )
 .withColumn(
     'position',
     F.when(F.col('team') == 'football','football')
     .when(F.col('position') == 'FB','RB')
     .otherwise(F.col('position'))
 )
)

#%%

min_frames = (tracking_w_side
 .groupBy(F.col('gameId'),F.col('playId'))
 .agg(F.min(F.col('frameId')).alias('min_frame'))
 )

spots = (tracking_w_side
 .join(
     min_frames,
     ['gameId','playId']
    )
 .filter((F.col('team') == 'football') & (F.col('frameId') == F.col('min_frame')))
 .select('gameId','playId',F.col('x').alias('spot_x'),F.col('y').alias('spot_y'))
 )

tracking_flipped = (tracking_w_side
 .join(
     spots,
     ['gameId','playId'] 
    )
 .with_column('newx',F.when(F.col('playDirection') == 'right',F.col('y')).otherwise(-F.col('y') + 53.33))
 .with_column('newy',F.when(F.col('playDirection') == 'right',-F.col('x')).otherwise(F.col('x')))
 .with_column('newo',F.when(F.col('playDirection') == 'right',-(F.col('o') + 90)).otherwise(-(F.col('o') - 90)))
 .with_column('newspot_y',F.when(F.col('playDirection') == 'right',-F.col('spot_x')).otherwise(F.col('spot_x')))
 .with_column('newspot_x',F.when(F.col('playDirection') == 'right',F.col('spot_y')).otherwise( - F.col('spot_y') + 53.33))
 .with_column('line_to_gain',F.col('newspot_y') - F.col('yardsToGo'))
 .with_column('newy',F.when(F.col('playDirection') == 'right',F.col('newy') + 110).otherwise(F.col('newy') - 10))
 .with_column('newspot_y',F.when(F.col('playDirection') == 'right',F.col('newspot_y') + 110).otherwise(F.col('newspot_y') - 10))
 .with_column('line_to_gain',F.when(F.col('playDirection') == 'right',F.col('line_to_gain') + 110).otherwise(F.col('line_to_gain') - 10))
 .select(
     'season','gameId','playId','frameId','team','nflId','defensiveTeam',
     F.col('newx').alias('x'),F.col('newy').alias('y'),F.col('newo').alias('o'),
     F.col('newspot_y').alias('spot_y'),F.col('newspot_x').alias('spot_x'),'line_to_gain',
     's','a','dis','dir',
     'yardsToGo','down','side','position'
    )
 .filter(F.col('spot_y') >= 30)
 .with_column('depth',F.col('y') - F.col('spot_y'))
 .with_column('width',F.col('x') - F.col('spot_x'))
)

deep7 = (tracking_flipped
 .filter(F.col('side') == "defense")
 .groupBy('gameId','playId','nflId')
 .agg(F.mean('depth').alias('depth'))
 .with_column('depth_rk',F.dense_rank().over(window.partition_by(F.col('gameId'),F.col('playId')).order_by(F.col('depth'))))
 .filter(F.col('depth_rk') <= 7)
 )

tracking_deep7 = tracking_flipped.join(deep7,['gameId','playId','nflId'],'semi')

#%%

skill_position_ids = (tracking_w_side
 .select('gameId','playId','nflId','season','side','position')
 .drop_duplicates()
 .filter(F.col('side') == "offense")
 .filter(F.col('position').in_(['QB','RB','WR','TE']))
 )

tracking_skill6 = (tracking_flipped
 .join(
     skill_position_ids,
     ['gameId','playId','nflId'],
     'semi'
    )
)

#%%

coverages_to_model = ['Cover-1', 'Cover-3', 
                      'Cover-6', 'Quarters', 
                      'Cover-2', '2-Man']

coverage_hierarchies = pd.DataFrame({
    'PFF_PASSCOVERAGE':coverages_to_model,
    'HIERARCHY':[1,1,0,0,0,0]
})

coverage_hierarchies = session.write_pandas(
    coverage_hierarchies,
    'coverage_hierarchies',
    auto_create_table = True,
    table_type = 'temp',
    overwrite = True
)

coverage_mapping = (plays
 .filter(F.col('pff_passCoverage').in_(coverages_to_model))
 .select('gameId','playId','pff_passCoverage')
 .groupBy('pff_passCoverage')
 .agg(F.count('*').alias('count'))
 .with_column('coverage',F.dense_rank().over(Window.order_by(F.col('count'))))
 .with_column('coverage',F.col('coverage') - 1)
)

coverages_by_clip = (plays
 .join(
     coverage_mapping,
     'pff_passCoverage'
    )
 .join(
     coverage_hierarchies,
     'pff_passCoverage' 
    )
 .select('gameId','playId','coverage','hierarchy')
 )

#%%

clip_spec = Window.order_by(F.col('gameId'),F.col('playId'))

plays_w_clip_id = (plays
 .with_column('clip_id',F.dense_rank().over(clip_spec))
 [['gameId','playId','clip_id']]
)

offense_defense = (tracking_deep7
 .union(tracking_skill6)
  .join(
     coverages_by_clip,
     ['gameId','playId']
    )
 .join(
     plays_w_clip_id,
     ['gameId','playId']
    )
)

players_by_clip = (offense_defense
 .groupBy('clip_id')
 .agg(F.count_distinct(F.col("nflId")).alias('players'))
 .filter(F.col('players') == 13)
 )

model_df = offense_defense.join(players_by_clip,'clip_id','semi')

#%%

from sklearn.model_selection import train_test_split

clip_ids = (model_df
 .select('clip_id','coverage','hierarchy')
 .drop_duplicates()
 .to_pandas()
)

train_clips, test_clips = train_test_split(clip_ids, test_size = 0.2, random_state = 42, stratify = clip_ids['COVERAGE'])
ana_clips, assess_clips = train_test_split(train_clips,test_size = .25, random_state = 15, stratify = train_clips['COVERAGE'])

#%%

all_sets = pd.concat([
    ana_clips.assign(SET = 'ana'),
    assess_clips.assign(SET = 'assess'),
    test_clips.assign(SET = 'test')
])

all_sets_w_play_ids = (all_sets
 .merge(
     plays_w_clip_id.to_pandas(),
     on = 'CLIP_ID',
     how = 'left'
    )
)

all_sets_table = session.write_pandas(
    all_sets_w_play_ids,
    'SET_SPLIT',
    auto_create_table = True,
    overwrite = True
)

model_df.write.save_as_table(
    "MODEL_DF", 
    mode = "overwrite"
)

coverage_mapping.write.save_as_table(
    "COVERAGE_MAPPING", 
    mode = "overwrite"
)