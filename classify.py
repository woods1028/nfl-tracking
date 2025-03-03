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
    batch_size = 32, 
    mask_value = mask_value, 
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

one_high_preds = hierarchy_preds.query('pred > .5')

one_high_clips = one_high_preds['clip_id']

one_high_model = gru_model_w_mask(
    transition_feature_dim = 14,
    defender_feature_dim = 42,
    offense_feature_dim = 36,
    mask_value = mask_value
)

one_high_model, one_high_history, one_high_preds = model_train(
    session, 
    model, 
    batch_size = 32, 
    mask_value = mask_value, 
    num_epochs = 30
)

