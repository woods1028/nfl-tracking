#%%

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from features import process_play_data_w_mask

#%%

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from snowflake.snowpark import functions as F

#%%

def get_model_inputs_cv(session, model_df, subset, label_col, mask_value):

    if subset is not None:

        model_df = (model_df
         .join(
             subset,
             on = 'CLIP_ID',
             how = 'inner'
            )
        )

    clip_ids_labels = (model_df
     .select('clip_id',label_col)
     .drop_duplicates()
     .to_pandas()
     .sort_values('CLIP_ID')
     .values 
    ) 

    clip_ids = [x[0] for x in clip_ids_labels]
    labels = [x[1] for x in clip_ids_labels]

    model_inputs = process_play_data_w_mask(
        session, model_df, labels, mask_value = mask_value
    )

    model_inputs = list(model_inputs)

    model_inputs.append(np.array(clip_ids))

    return model_inputs

def model_train_cv(session, fresh_model, model_df, set_split, subset, label_col, mask_value, batch_size, num_epochs):

    model_inputs = get_model_inputs_cv(
        session, model_df, subset, label_col, mask_value
    )

    clip_ids_by_fold = set_split.to_pandas()[['FOLD','CLIP_ID']].to_numpy()

    folds = np.unique([x[0] for x in clip_ids_by_fold])[:-1]

    histories = []
    fold_models = []

    for fold in folds:

        print(f'Training fold {fold}...')

        fold_clip_ids = [x[1] for x in clip_ids_by_fold if x[0] == fold] 

        fold_indices = [x in fold_clip_ids for x in model_inputs[5]]
        not_fold_indices = [not x for x in fold_indices]

        assess_inputs = [x[fold_indices] for x in model_inputs]
        ana_inputs = [x[not_fold_indices] for x in model_inputs]

        fold_model = fresh_model()

        # Train model
        history = fold_model.fit(
            ana_inputs[0:4],
            ana_inputs[4],
            validation_data=(assess_inputs[0:4],assess_inputs[4]),
            #class_weight = class_weights_dict,
            epochs = num_epochs,
            batch_size = batch_size,
            verbose=1
        )

        histories.append(history)
        fold_models.append(fold_model)

    fold_preds = [model.predict(model_inputs[0:4]) for model in fold_models]

    pred_col_names = ['pred' + str(x) for x in range(len(fold_preds[0][0]))]

    preds = pd.concat([
        (pd.DataFrame(y,columns = pred_col_names)
         .assign(
             fold = z,
             actual = model_inputs[4],
             clip_id = model_inputs[5]
            )
        ) for y, z in zip(fold_preds,range(len(fold_preds)))
    ])

    if (len(pred_col_names) == 1):

        preds = preds.rename(columns = {'pred0':'pred1'}).assign(pred0 = lambda x: 1 - x['pred1'])

    return fold_models, histories, preds

#%%

def eval_cv(histories, preds, return_type, coverage_mapping):

    pred_columns = [x for x in preds.columns if 'pred' in x]

    preds_categorical = (preds
     .melt(
         id_vars = ['fold','actual','clip_id'],
         var_name = 'class',
         value_name = 'pct',
         value_vars = pred_columns
        )
     .assign(
         pred_coverage = lambda x: x['class'].str.replace('pred','').astype(int),
         max_pct = lambda x: x.groupby('clip_id')['pct'].transform('max'),
         pred = lambda x: x.apply(lambda x: x['pred_coverage'] if x['pct'] == x['max_pct'] else np.nan, axis = 1)
        )
     .merge(
         coverage_mapping.rename(columns = {'coverage':'actual','class':'actual_class'}),
         on = 'actual'
        ) 
     .merge(
         coverage_mapping.rename(columns = {'coverage':'pred','class':'pred_class'}),
         on = 'pred'
        ) 
     [['fold','clip_id','actual','pred_coverage','pred','pct','actual_class','pred_class']]
     ) 
    
    if return_type == 'accuracy matrix':

        accuracy_matrix = (preds_categorical
         .assign(match = lambda x: x['pred'] == x['actual'])
         .groupby(['fold','pred_class'])
         .agg(count = ('clip_id','size'),
              correct = ('match','sum'))
         .reset_index()
         .assign(pct = lambda x: round(x['correct']/x['count'],2))
         )

        return accuracy_matrix
    
    if return_type == 'density':

        g = sns.FacetGrid(
            preds_categorical.assign(pred_class = lambda x: x['pred_class'].astype('category')), 
            col = "actual_class", 
            col_wrap = len(pred_columns), 
            height=4,
            sharey = False
        )

        g.map_dataframe(sns.kdeplot,x = 'pct',hue = 'pred_class',alpha = .5, fill = True)

        # Manually create a legend using Seaborn color palette
        palette = sns.color_palette("tab10", n_colors=preds_categorical["pred_class"].nunique())
        legend_labels = preds_categorical.assign(pred_class = lambda x: x['pred_class'].astype('category'))["pred_class"].cat.categories
        legend_handles = [plt.Line2D([0], [0], color=palette[i], lw=4, label=label) for i, label in enumerate(legend_labels)]

        # Add Custom Legend
        g.figure.legend(handles=legend_handles, title="Prediction Class", loc="center right", frameon=False,bbox_to_anchor=(1.05, 1))  # Moves legend outside)

        plt.show()

    if return_type == 'history':

        history_metrics = []

        for history, fold in zip(histories,range(len(histories))):

            model_metrics = [history.history[x] for x in history.history.keys()]

            metrics_df = (pd.concat(
                 [pd.Series(col) for col in model_metrics],
                 axis = 1
                )
             .set_axis(list(history.history.keys()),axis = 1)
             .assign(epoch = lambda x: [y + 1 for y in range(len(x))])
             .melt(id_vars = 'epoch')
             .assign(set = lambda x: x['variable'].apply(lambda x: 'assess' if 'val' in x else 'ana'),
                     metric = lambda x: x['variable'].apply(lambda x: 'accuracy' if 'accuracy' in x else 'loss'),
                     fold = fold)
            )

            history_metrics.append(metrics_df)

        history_metrics = pd.concat(history_metrics)

        # Create FacetGrid
        g = sns.FacetGrid(history_metrics, col="metric", row = 'fold')

        # Map the line plot to each facet
        g.map_dataframe(sns.lineplot, "epoch", "value",hue = 'set')

        g.add_legend() 

        plt.show()

    if return_type == "confusion matrix":

        confusion_matrix = (preds_categorical
         .groupby(['fold','actual_class','pred_class'])
         .agg(count = ('clip_id','size'))
         .reset_index()
         .assign(pct = lambda x: x['count']/x.groupby(['fold','actual_class'])['count'].transform('sum'))
         .assign(pct = lambda x: x['pct'].round(2))
         .pivot(index = ['fold','actual_class'],columns = 'pred_class',values = 'pct')
         .reset_index()
         .fillna(0)
        )

        return confusion_matrix



    

