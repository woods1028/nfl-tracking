#%%

import pandas as pd
import datetime
from functools import partial
import snowflake.snowpark.functions as F
from snwflk import snowflake_connect
from gru_models import gru_model_w_mask, gru_model_w_static
from model import model_train, model_train_cv, eval, eval_cv

#%%

session = snowflake_connect()

set_split = session.table('set_split')

model_df = (session.table("model_df")
 .join(
     set_split.select('clip_id','set'),
     'clip_id'
    )
)

coverage_mapping = session.table('coverage_mapping').to_pandas()

#%% 
"""
CV for 2-class hierarchical model
"""

hierarchy_mapping = pd.DataFrame({
    'coverage':[0,1],
    'class':['2-High','1-High']
})

mask_value = -9999

hierarchy_model = gru_model_w_static(
    transition_feature_dim = 14,
    defender_feature_dim = 42,
    offense_feature_dim = 36,
    static_shape = (3,),
    mask_value = mask_value,
    model_type = 'sigmoid',
    num_classes = 2
)

hierarchy_model = partial(
    gru_model_w_static,
    transition_feature_dim = 14,
    defender_feature_dim = 42,
    offense_feature_dim = 36,
    static_shape = (3,),
    mask_value = mask_value,
    model_type = 'sigmoid',
    num_classes = 2
)

hierarchy_models, hierarchy_histories, hierarchy_preds = model_train_cv(
    session, 
    hierarchy_model, 
    model_df.filter(F.col('set').in_(['ana','assess'])),
    set_split,
    None, 
    'hierarchy', 
    mask_value, 
    batch_size = 32, 
    num_epochs = 30
)

eval_cv(hierarchy_histories, hierarchy_preds, 'accuracy matrix', hierarchy_mapping)

eval_cv(hierarchy_histories, hierarchy_preds, 'density', hierarchy_mapping)

eval_cv(hierarchy_histories, hierarchy_preds, 'history', hierarchy_mapping)

eval_cv(hierarchy_histories, hierarchy_preds, 'confusion matrix', hierarchy_mapping)

#%% 

"""
Full 2-class hierarchical model
"""

mask_value = -9999

hierarchy_model_full = gru_model_w_static(
    transition_feature_dim = 14,
    defender_feature_dim = 42,
    offense_feature_dim = 36,
    static_shape = (3,),
    mask_value = mask_value,
    model_type = 'sigmoid',
    num_classes = 2
)

hierarchy_model_full, hierarchy_history_full, hierarchy_preds_full = model_train(
    session = session, 
    model = hierarchy_model_full, 
    model_df = model_df,
    train_set = ['assess','ana'],
    test_set = 'test',
    subset = None, 
    label_col = 'hierarchy', 
    class_weighting = None,
    mask_value = mask_value, 
    batch_size = 32, 
    num_epochs = 30
)

#%%

eval(hierarchy_history_full, hierarchy_preds_full, 'accuracy matrix', hierarchy_mapping)

eval(hierarchy_history_full, hierarchy_preds_full, 'density', hierarchy_mapping)

eval(hierarchy_history_full, hierarchy_preds_full, 'history', hierarchy_mapping)

eval(hierarchy_history_full, hierarchy_preds_full, 'bin accuracy', hierarchy_mapping)

#%%

hierarchy_model_full.save('hierarchy_model.keras')

# (hierarchy_preds
#  .rename(columns = {'clip_id':'CLIP_ID'})
#  .merge(
#      set_split.to_pandas(),
#      on = 'CLIP_ID'
#     )
#  .to_csv("all_preds_" + datetime.datetime.now().strftime("%Y%m%d") + ".csv",index = False)
# )

#%%

one_high_mapping = (hierarchy_preds
 .query('pred > .5')
 .assign(CLASS = lambda x: x['COVERAGE'].map({0:0,1:0,2:0,3:0,4:1,5:2}))
 .merge(
     coverage_mapping,
     on = 'COVERAGE'
    )
)

(one_high_mapping
 .groupby('CLASS')
 .agg(coverages = ('PFF_PASSCOVERAGE',lambda x: '; '.join(x.unique())))
 .reset_index()
 .to_csv("one_high_mapping.csv",index = False)
)

one_high_subset = one_high_mapping[['CLIP_ID','CLASS']]

#%%

one_high_subset

mask_value = -9999

one_high_model = gru_model_w_mask(
    transition_feature_dim = 14,
    defender_feature_dim = 42,
    offense_feature_dim = 36,
    mask_value = mask_value,
    model_type = 'softmax', 
    num_classes = 3
)

one_high_model, one_high_history, one_high_preds = model_train(
    session, 
    model = one_high_model, 
    subset = one_high_subset, 
    label_col = 'class', 
    mask_value = mask_value, 
    batch_size = 32, 
    num_epochs = 30
)

#%%

eval(one_high_history, one_high_preds, 'accuracy matrix')

eval(one_high_history, one_high_preds, 'history')

eval(one_high_history, one_high_preds, 'bin accuracy')

#%%

one_high_model.save("one_high_model.keras")

(one_high_preds
 .rename(columns = {'clip_id':'CLIP_ID'})
#  .merge(
#      set_split.to_pandas(),
#      on = 'CLIP_ID'
#     )
 .to_csv("one_high_preds_20250302.csv",index = False)
)

#one_high_preds = pd.read_csv("one_high_preds_20250302.csv")

#%%

two_high_subset = session.create_dataframe(
    (hierarchy_preds
     .query('pred < .5')
     .rename(columns = lambda x: x.upper())
     .assign(CLASS = lambda x: x['COVERAGE'].map({0:0,1:1,2:2,3:3,4:4,5:4}))
     [['CLIP_ID','CLASS']]
    )
)

two_high_mapping = (hierarchy_preds
 .query('pred <= .5')
 .assign(CLASS = lambda x: x['COVERAGE'].map({0:0,1:1,2:2,3:3,4:4,5:4}))
 .merge(
     coverage_mapping,
     on = 'COVERAGE'
    )
)

(two_high_mapping
 .groupby('CLASS')
 .agg(coverages = ('PFF_PASSCOVERAGE',lambda x: '; '.join(x.unique())))
 .reset_index()
 .to_csv("two_high_mapping.csv",index = False)
)

two_high_subset = two_high_mapping[['CLIP_ID','CLASS']]

#%%

mask_value = -9999

two_high_model = gru_model_w_mask(
    transition_feature_dim = 14,
    defender_feature_dim = 42,
    offense_feature_dim = 36,
    mask_value = mask_value,
    model_type = 'softmax', 
    num_classes = 5
)

two_high_model, two_high_history, two_high_preds = model_train(
    session, 
    model = two_high_model, 
    subset = two_high_subset, 
    label_col = 'class', 
    mask_value = mask_value, 
    batch_size = 32, 
    num_epochs = 30
)

#%%

eval(two_high_history, two_high_preds, 'accuracy matrix')

eval(two_high_history, two_high_preds, 'history')

eval(two_high_history, two_high_preds, 'bin accuracy')

#%%

two_high_model.save("two_high_model.keras")

(two_high_preds
 .rename(columns = {'clip_id':'CLIP_ID'})
 .merge(
     set_split.to_pandas(),
     on = 'CLIP_ID'
    )
 .to_csv("two_high_preds_20250302.csv",index = False)
)

#%%

all_class_mapping = (model_df
 .filter(F.col('coverage') != 0)
 .with_column('class',F.col('coverage') - 1)            
 )

all_class_model_mapping = (all_class_mapping
 .select('coverage','class')
 .drop_duplicates()
 .to_pandas()
 .merge(
     coverage_mapping,
     on = 'COVERAGE'
    )
 .rename(columns = {'PFF_PASSCOVERAGE':'coverages'})
 [['CLASS','coverages']]
 )

all_class_model_mapping.to_csv("all_class_mapping.csv",index = False)

all_class_subset = (all_class_mapping
 .select('clip_id','class')
 .drop_duplicates()
)

#%%

all_class_model = gru_model_w_static(
    transition_feature_dim = 14,
    defender_feature_dim = 42,
    offense_feature_dim = 36,
    mask_value = mask_value,
    static_shape = (3,),
    model_type = 'softmax', 
    num_classes = 5
)   

all_class_model, all_class_history, all_class_preds = model_train(
    session = session,
    model = all_class_model, 
    model_df = model_df,
    subset = all_class_subset, 
    label_col = 'class', 
    mask_value = mask_value, 
    batch_size = 32, 
    num_epochs = 50
)

#%%

eval(all_class_history, all_class_preds, 'accuracy matrix', coverage_mapping).query('set == "assess"')

eval(all_class_history, all_class_preds, 'history', coverage_mapping)

eval(all_class_history, all_class_preds, 'confusion matrix', all_class_model_mapping)

#%%

all_class_preds.to_csv("all_class_preds_20250306.csv",index = False)

#%% other way to do the hierarchy

hierarchy3_mapping = (model_df
 .with_column('hierarchy',F.when(F.col('coverage').in_([1,2]),'2').otherwise(F.col('hierarchy')))
 )

hierarchy3_model = gru_model_w_static(
    transition_feature_dim = 14,
    defender_feature_dim = 42,
    offense_feature_dim = 36,
    mask_value = mask_value,
    static_shape = (3,),
    model_type = 'softmax', 
    num_classes = 3
)   

hierarchy3_model, hierarchy3_history, hierarchy3_preds = model_train(
    session = session,
    model = hierarchy3_model, 
    model_df = hierarchy3_mapping,
    subset = None, 
    label_col = 'hierarchy', 
    mask_value = mask_value, 
    batch_size = 32, 
    num_epochs = 50
)

#%%

hierarchy_mapping = pd.DataFrame({
    'coverage':[0.0,1.0,2.0],
    'class':['2-High','1-High','Quarters']
})

eval(hierarchy3_history, hierarchy3_preds, 'accuracy matrix', hierarchy_mapping).query('set == "assess"')

eval(hierarchy3_history, hierarchy3_preds, 'history', hierarchy_mapping)

eval(hierarchy3_history, hierarchy3_preds, 'confusion matrix', hierarchy_mapping)
