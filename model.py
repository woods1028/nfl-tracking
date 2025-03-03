#%%

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from snwflk import snowflake_connect
from gru_models import gru_model_w_mask
from features import process_play_data_w_mask

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from snowflake.snowpark import functions as F

#%%

def get_model_inputs(session, label_col, mask_value):

    session = snowflake_connect()

    set_split = session.table('set_split')

    model_df = (session.table("model_df")
     .join(
        set_split.select('clip_id','set'),
        'clip_id'
        )
     .filter(F.col('set').in_(['ana','assess']))
    )

    ana_set_df = model_df.filter(F.col('set') == "ana")
    ana_clip_ids_labels = ana_set_df.select('clip_id',label_col).drop_duplicates().to_pandas().sort_values('CLIP_ID').values    
    ana_clip_ids = [x[0] for x in ana_clip_ids_labels]
    ana_labels = [x[1] for x in ana_clip_ids_labels]

    assess_set_df = model_df.filter(F.col('set') == "assess")
    assess_clip_ids_labels = assess_set_df.select('clip_id',label_col).drop_duplicates().to_pandas().sort_values('CLIP_ID').values
    assess_clip_ids = [x[0] for x in assess_clip_ids_labels]
    assess_labels = [x[1] for x in assess_clip_ids_labels]

    print("Gathering transition/defender data... ")

    ana_transition_data, ana_defender_data, ana_offense_data, ana_processed_labels = process_play_data_w_mask(
        session, ana_set_df, ana_labels, mask_value = mask_value
    )

    assess_transition_data, assess_defender_data, assess_offense_data, assess_processed_labels = process_play_data_w_mask(
        session, assess_set_df, assess_labels, mask_value = mask_value
    )

    model_inputs = [
        [[ana_transition_data, ana_defender_data, ana_offense_data], ana_processed_labels, ana_clip_ids],
        [[assess_transition_data, assess_defender_data, assess_offense_data], assess_processed_labels, assess_clip_ids]
    ]

    print("Model inputs gathered.")

    return model_inputs

def model_train(session, batch_size, mask_value, num_epochs):

    ana_model_inputs, assess_model_inputs = get_model_inputs(session)

    model = gru_model_w_mask(
        transition_feature_dim = 14,
        defender_feature_dim = 42,
        offense_feature_dim = 36,
        mask_value = mask_value
    )

    # Train model
    history = model.fit(
        ana_model_inputs[0],
        ana_model_inputs[1],
        validation_data=(assess_model_inputs[:-1]),
        epochs = num_epochs,
        batch_size = batch_size,
        verbose=1
    )

    train_preds = (pd.DataFrame(model.predict(ana_model_inputs[0]),columns = ['pred'])
     .assign(
         set = 'ana',
         actual = ana_model_inputs[1],
         clip_id = ana_model_inputs[2]
        )
    )

    test_preds = (pd.DataFrame(model.predict(assess_model_inputs[0]),columns = ['pred'])
     .assign(
         set = 'assess',
         actual = assess_model_inputs[1],
         clip_id = assess_model_inputs[2]
        )
    )

    preds = pd.concat([train_preds,test_preds])

    return model, history, preds

#%%

def eval(history, preds, return_type):

    if return_type == 'accuracy matrix':

        accuracy_matrix = (preds
        .assign(pred = lambda x: x['pred'].apply(lambda x: 1 if x > .5 else 0),
                match = lambda x: x['pred'] == x['actual'])
        .groupby(['set','pred'])
        .agg(count = ('clip_id','size'),
            correct = ('match','sum'))
        .reset_index()
        .assign(pct = lambda x: round(x['correct']/x['count'],2))
        )

        return accuracy_matrix
    
    if return_type == 'density':

        sns.kdeplot(
            data = (preds
            .assign(
                pred_binary = lambda x: x['pred'].apply(lambda x: 1 if x > .5 else 0),
                coverage = lambda x: x['actual'].map({1:'one high',0:'two high'})
                )
            .assign(coverage = lambda x: x.apply(lambda x: x['coverage'] if x['actual'] == x['pred_binary'] else 'wrong',axis = 1))
            ), 
            x='pred', 
            hue='coverage', 
            fill=True, 
            alpha=0.5
        )

        plt.title("Confusion Matrix Density")
        plt.grid(True) 
        plt.show()

    if return_type == 'history':

        model_metrics = [history.history[x] for x in history.history.keys()]

        metrics_df = (pd.concat(
            [pd.Series(col) for col in model_metrics],
            axis = 1
            )
        .set_axis(list(history.history.keys()),axis = 1)
        .assign(epoch = lambda x: [y + 1 for y in range(len(x))])
        .melt(id_vars = 'epoch')
        .assign(set = lambda x: x['variable'].apply(lambda x: 'assess' if 'val' in x else 'ana'),
                metric = lambda x: x['variable'].apply(lambda x: 'accuracy' if 'accuracy' in x else 'loss'))
        )

        # Create FacetGrid
        g = sns.FacetGrid(metrics_df, col="metric", col_wrap=2, height=4)

        # Map the line plot to each facet
        g.map_dataframe(sns.lineplot, "epoch", "value",hue = 'set')

        g.add_legend() 

        # Show plot
        plt.grid(True) 
        plt.show()

    if return_type == 'bin accuracy':

        bins = [x/10 for x in range(11)]
        bin_labels = [x/100 for x in range(5,105,10)]

        bin_metrics = (preds
         .assign(bin = lambda x: pd.cut(
             x['pred'],
             bins = bins,
             labels = bin_labels
            ))
         .assign(bin = lambda x: x['bin'].astype(float))
         .groupby(['set','actual','bin'])
         .agg(count = ('clip_id','size'))
         .reset_index()
         .assign(pct = lambda x: x['count']/x.groupby(['set','bin'])['count'].transform('sum'))
         .query('actual == 1')
        )

        sns.scatterplot(
            data = bin_metrics,
            x = 'bin',
            y = 'pct',
            hue = 'set',
            size = 'count'
        )

        plt.title("Bin Accuracy")
        plt.xticks(bin_labels)
        plt.yticks(bin_labels)
        plt.grid(True) 
        plt.show()

    return []


    

