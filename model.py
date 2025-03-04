import numpy as np

#%%

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from features import process_play_data_w_mask

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from snowflake.snowpark import functions as F

#%%

def get_model_inputs(session, subset, label_col, mask_value):

    set_split = session.table('set_split')

    model_df = (session.table("model_df")
     .join(
        set_split.select('clip_id','set'),
        'clip_id'
        )
     .filter(F.col('set').in_(['ana','assess']))
    )

    if subset is not None:

        model_df = (model_df
         .join(
             subset,
             on = 'CLIP_ID',
             how = 'inner'
            )
         )

    ana_set_df = model_df.filter(F.col('set') == "ana")
    assess_set_df = model_df.filter(F.col('set') == "assess")

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

def model_train(session, model, subset, label_col, mask_value, batch_size, num_epochs):

    ana_model_inputs, assess_model_inputs = get_model_inputs(
        session,
        subset, 
        label_col, 
        mask_value
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

    return model, history, preds

#%%

def eval(history, preds, return_type):

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
         pred_class = lambda x: x['class'].str.replace('pred','').astype(int),
         max_pct = lambda x: x.groupby('clip_id')['pct'].transform('max'),
         pred = lambda x: x.apply(lambda x: x['pred_class'] if x['pct'] == x['max_pct'] else np.nan, axis = 1)
        )
     [['set','clip_id','actual','pred_class','pred','pct']]
     )
    
    #%%

    if return_type == 'accuracy matrix':

        accuracy_matrix = (preds
         .merge(
             preds_categorical.query('pred == pred')[['clip_id','pred']],
             on = 'clip_id'
            )
         .assign(match = lambda x: x['pred'] == x['actual'])
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

        #%%

        bin_metrics = (preds
         .merge(
             preds_categorical[['clip_id','pred_class','pred','pct']],
             on = 'clip_id'
            )
         .assign(match = lambda x: x['actual'] == x['pred'])
         .assign(bin = lambda x: pd.cut(
             x['pct'],
             bins = bins,
             labels = bin_labels
            ))
         .assign(bin = lambda x: x['bin'].astype(float))
         .groupby(['set','actual','bin'])
         .agg(count = ('clip_id','nunique'),
              correct = ('match','sum'))
         .reset_index()
         .assign(pct = lambda x: x['correct']/x['count'])
         .query('bin > .5 and pct != 0')
        )

        #%%

        g = sns.FacetGrid(bin_metrics, col="actual", col_wrap = len(pred_columns), height=4)

        g.map_dataframe(sns.scatterplot,x = 'bin',y = 'pct',hue = 'set',size = 'count')

        g.add_legend() 

        for ax in g.axes.flat:
            ax.grid(True, linewidth = 0.5)
            ax.set_xticks(bin_labels[5:])
            ax.set_yticks(bin_labels[5:])

        handles, labels = g.legend.legend_handles, [t.get_text() for t in g._legend.texts]
        g._legend.remove()  # Remove the default legend

        g.figure.legend(handles[1:3], labels[1:3], title="set", loc="center right", frameon=False)

        plt.show()

        #%%

    if return_type == 'confusion matrix':

        confusion_matrix = (preds_categorical
         .groupby(['actual','pred'])
         .agg(
             count = ('clip_id','size'),
            )
         .reset_index()
         .assign(pct = lambda x: x['count']/x.groupby('actual')['count'].transform('sum'))
         .assign(pct = lambda x: x['pct'].round(2))
         .pivot(index = 'actual',columns = 'pred',values = 'pct')
        )

        return confusion_matrix

    return None


    

