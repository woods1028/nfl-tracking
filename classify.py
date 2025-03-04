#%%

from snwflk import snowflake_connect
from gru_models import gru_model_w_mask
from model import model_train, eval

#%%

session = snowflake_connect()

mask_value = -9999

model = gru_model_w_mask(
    transition_feature_dim = 14,
    defender_feature_dim = 42,
    offense_feature_dim = 36,
    mask_value = mask_value
)

hierarchy_model, hierarchy_history, hierarchy_preds = model_train(
    session, 
    model, 
    None, 
    'hierarchy', 
    mask_value, 
    batch_size = 32, 
    num_epochs = 30
)

#%%

eval(hierarchy_history, hierarchy_preds , 'accuracy matrix')

eval(hierarchy_history, hierarchy_preds, 'density')

eval(hierarchy_history, hierarchy_preds, 'history') ## need to flip it off after 30

eval(hierarchy_history, hierarchy_preds, 'bin accuracy')

#%%

model.save('hierarchy_model.keras')

#%%

# (preds
#  .rename(columns = {'clip_id':'CLIP_ID'})
#  .merge(
#      set_split.to_pandas(),
#      on = 'CLIP_ID'
#     )
#  .to_csv("all_preds_20250302.csv",index = False)
# )

#%%

one_high_subset = session.create_dataframe(
    (hierarchy_preds
     .query('pred > .5')
     .rename(columns = lambda x: x.upper())
     .merge(
         set_split.to_pandas(),
         on = ['CLIP_ID','SET']
     )
     .assign(CLASS = lambda x: x['COVERAGE'].map({0:0,1:0,2:0,3:0,4:1,5:2}))
     [['CLIP_ID','CLASS']]
    )
)

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
 .merge(
     set_split.to_pandas(),
     on = 'CLIP_ID'
    )
 .to_csv("one_high_preds_20250302.csv",index = False)
)

#%%

two_high_subset = session.create_dataframe(
    (hierarchy_preds
     .query('pred < .5')
     .rename(columns = lambda x: x.upper())
     .merge(
         set_split.to_pandas(),
         on = ['CLIP_ID','SET']
     )
     .assign(CLASS = lambda x: x['COVERAGE'].map({0:0,1:1,2:2,3:3,4:4,5:4}))
     [['CLIP_ID','CLASS']]
    )
)

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

