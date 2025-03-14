#%%

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

from features import process_play_data_w_mask

#%%

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from snowflake.snowpark import functions as F

#%%

def get_model_inputs(session, model_df, subset, train_set, test_set, label_col, mask_value):

    if subset is not None:

        model_df = (model_df
         .join(
             subset,
             on = 'CLIP_ID',
             how = 'inner'
            )
         )

    ana_set_df = model_df.filter(F.col('set').in_(train_set))
    assess_set_df = model_df.filter(F.col('set').in_(test_set))

    ana_clip_ids_labels, assess_clip_ids_labels = [(df
     .select('clip_id',label_col)
     .drop_duplicates()
     .to_pandas()
     .sort_values('CLIP_ID')
     .values 
    ) for df in [ana_set_df, assess_set_df]]

    ana_clip_ids = [x[0] for x in ana_clip_ids_labels]
    ana_labels = [x[1] for x in ana_clip_ids_labels]

    assess_clip_ids = [x[0] for x in assess_clip_ids_labels]
    assess_labels = [x[1] for x in assess_clip_ids_labels]

    print("Gathering transition/defender data... ")

    ana_transition_data, ana_defender_data, ana_offense_data, ana_static_data, ana_processed_labels = process_play_data_w_mask(
        session, ana_set_df, ana_labels, mask_value = mask_value
    )

    assess_transition_data, assess_defender_data, assess_offense_data, assess_static_data, assess_processed_labels = process_play_data_w_mask(
        session, assess_set_df, assess_labels, mask_value = mask_value
    )

    model_inputs = [
        [[ana_transition_data, ana_defender_data, ana_offense_data, ana_static_data], ana_processed_labels, ana_clip_ids],
        [[assess_transition_data, assess_defender_data, assess_offense_data, assess_static_data], assess_processed_labels, assess_clip_ids]
    ]

    print("Model inputs gathered.")

    return model_inputs

def model_train(session, model, model_df, subset, train_set, test_set, label_col, class_weighting, mask_value, batch_size, num_epochs):

    ana_model_inputs, assess_model_inputs = get_model_inputs(
        session,
        model_df,
        subset, 
        train_set, test_set,
        label_col, 
        mask_value
    )

    if class_weighting != None:

        class_weights = compute_class_weight(
            class_weight= 'balanced', 
            classes = np.unique(ana_model_inputs[1]), 
            y = ana_model_inputs[1]
        )
        
        class_weights_dict = dict(zip(np.unique(ana_model_inputs[1]),class_weights))

    else:

        class_weights_dict = None

    # Train model
    history = model.fit(
        ana_model_inputs[0],
        ana_model_inputs[1],
        validation_data = (assess_model_inputs[:-1]),
        class_weight = class_weights_dict,
        epochs = num_epochs,
        batch_size = batch_size,
        verbose=1
    )

    train_preds, test_preds = [model.predict(x) for x in [ana_model_inputs[0],assess_model_inputs[0]]]

    pred_col_names = ['pred' + str(x) for x in range(len(train_preds[0]))]

    train_preds = (pd.DataFrame(train_preds,columns = pred_col_names)
     .assign(
         set = 'ana',
         actual = ana_model_inputs[1],
         clip_id = ana_model_inputs[2]
        )
    )

    test_preds = (pd.DataFrame(test_preds,columns = pred_col_names)
     .assign(
         set = 'assess',
         actual = assess_model_inputs[1],
         clip_id = assess_model_inputs[2]
        )
    )

    preds = pd.concat([train_preds,test_preds])

    if (len(pred_col_names) == 1):

        preds = preds.rename(columns = {'pred0':'pred1'}).assign(pred0 = lambda x: 1 - x['pred1'])

    return model, history, preds

#%%

def eval(history, preds, return_type, coverage_mapping):

    #%%

    pred_columns = [x for x in preds.columns if 'pred' in x]

    preds_categorical = (preds
     .melt(
         id_vars = ['set','actual','clip_id'],
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
     [['set','clip_id','actual','pred_coverage','pred','pct','actual_class','pred_class']]
     )
    
    #%%

    if return_type == 'accuracy matrix':

        accuracy_matrix = (preds
         .merge(
             preds_categorical.query('pred == pred')[['clip_id','pred']],
             on = 'clip_id'
            )
         .assign(match = lambda x: x['pred'] == x['actual'])
         .groupby(['set','pred'],dropna = False)
         .agg(count = ('clip_id','size'),
              correct = ('match','sum'))
         .reset_index()
         .assign(pct = lambda x: round(x['correct']/x['count'],2))
         .merge(
             coverage_mapping.rename(columns = {'coverage':'pred'}),
             on = ['pred'] 
            )
         [['set','class','count','correct','pct']]
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
        plt.show()

    if return_type == 'bin accuracy':

        bins = [x/10 for x in range(11)]
        bin_labels = [x/100 for x in range(5,105,10)]

        #%%

        bin_metrics = (preds
         .melt(
             id_vars = ['clip_id','set','actual'],
             value_vars = pred_columns,
             var_name = 'pred',
             value_name = 'pct'
            )
         .assign(pred = lambda x: x['pred'].str.replace('pred','').astype(int))
         .merge(
             coverage_mapping.rename(columns = {'coverage':'actual','class':'actual_class'}),
             on = 'actual'
            ) 
         .merge(
             coverage_mapping.rename(columns = {'coverage':'pred','class':'pred_class'}),
             on = 'pred'
            ) 
         .assign(match = lambda x: x['actual'] == x['pred'])
         .assign(bin = lambda x: pd.cut(
             x['pct'],
             bins = bins,
             labels = bin_labels
            ))
         .assign(bin = lambda x: x['bin'].astype(float))
         .groupby(['set','actual_class','bin'])
         .agg(count = ('clip_id','nunique'),
              correct = ('match','sum'))
         .reset_index()
         .assign(pct = lambda x: x['correct']/x['count'])
        )

        #%%

        g = sns.FacetGrid(bin_metrics, col="actual_class", col_wrap=len(pred_columns), height=4)

        g.map_dataframe(sns.scatterplot,x = 'bin',y = 'pct',hue = 'set',size = 'count')

        g.add_legend() 

        for ax in g.axes.flat:
            ax.grid(True, linewidth = 0.5)
            ax.set_xticks(bin_labels)
            ax.set_yticks(bin_labels)

        handles, labels = g.legend.legend_handles, [t.get_text() for t in g._legend.texts]
        g._legend.remove()  # Remove the default legend

        g.figure.legend(handles[1:3], labels[1:3], title="set", loc="center right", frameon=False)

        plt.show()

        #%%

    if return_type == 'confusion matrix':

        confusion_matrix = (preds_categorical
         .groupby(['actual','pred'])
         .agg(count = ('clip_id','size'))
         .reset_index()
         .assign(pct = lambda x: x['count']/x.groupby('actual')['count'].transform('sum'))
         .assign(pct = lambda x: x['pct'].round(2))
         .merge(coverage_mapping.rename(columns = {'coverage':'actual','class':'coverage_actual'})[['actual','coverage_actual']], on = 'actual')
         .merge(coverage_mapping.rename(columns = {'coverage':'pred','class':'coverage_pred'})[['pred','coverage_pred']], on = 'pred')
         .pivot(index = 'coverage_actual',columns = 'coverage_pred',values = 'pct')
         .reset_index()
        )

        return confusion_matrix

    return None



    

