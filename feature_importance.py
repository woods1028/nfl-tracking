#%%

import pandas as pd
import numpy as np
import re
from lime import lime_tabular

from keras.models import load_model
from model import get_model_inputs
from snwflk import snowflake_connect
from var_importance import predict_fn, flatten_dataset

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

hierarchy_model_full = load_model("hierarchy_model.keras")

train_model_inputs, test_model_inputs  = get_model_inputs(
    session, 
    model_df, 
    subset = None, 
    train_set = ['assess','ana'],
    test_set = 'test',
    label_col = 'hierarchy', 
    mask_value = -9999
)

#%% feature names

zone_order = ['blitz', 'flat_left', 'flat_right', 'hooks', 'middle', 'third_left', 'third_right']
transition_feature_names = [y + '__' + x for y in ['count','transition'] for x in zone_order]

tracking_features = ['x', 'y', 'dis', 's', 'o', 'dir']
defense_feature_names = ['defense' + str(i) + '__' + j for i in range(7) for j in tracking_features]
offense_feature_names = ['offense' + str(i) + '__' + j for i in range(6) for j in tracking_features]
static_feature_names = ['down','distance','spot']

feature_names = [x + '__frame' + str(i) for i in range(50) for x in transition_feature_names]
feature_names.extend([x + '__frame' + str(i) for i in range(51) for x in defense_feature_names])
feature_names.extend([x + '__frame' + str(i) for i in range(51) for x in offense_feature_names])
feature_names.extend(static_feature_names)

#%%

background_data = train_model_inputs[0]

test_examples = test_model_inputs[0]

background_flat = flatten_dataset(background_data)

test_example = [x[0] for x in test_examples]  # Get the first test sample
test_flat = np.concatenate([x.flatten() for x in test_example])

#%%

explainer = lime_tabular.LimeTabularExplainer(
    background_flat,
    feature_names = feature_names,
    class_names = ["2-High", "1-High"],  # Replace with your actual class names
    mode = "classification",
    random_state = 42
)

explanation = explainer.explain_instance(
    test_flat,
    predict_fn, 
    num_features = 20,  # Number of features to include in explanation
    top_labels = 1      # Explain top 1 predicted class
)

feature_importance = explanation.as_list()

#%%

metrics = ['mean','median','min','max','std']

(pd.DataFrame(feature_importance,columns = ['criterion','importance'])
 .assign(criterion = lambda x: x['criterion'].apply(lambda x: re.sub('[<,>]=* -*\\d+.\\d+','',x)))
 .assign(criterion = lambda x: x['criterion'].apply(lambda x: re.sub('-*\\d+.\\d+ [<,>]=* ','',x)))
 .assign(
     feature = lambda x: x['criterion'].str.split("__"),
     category = lambda x: x['feature'].apply(lambda x: x[0]),
     attribute = lambda x: x['feature'].apply(lambda x: x[1] if len(x) > 1 else '')
    )
 .groupby(['category','attribute'])
 .agg({**{'importance': metrics},})
 .reset_index()
 .sort_values(('importance','mean'),ascending = False)
)
