import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def vis_by_bar(vals, model_names=None, metric_names=None, **bar_param):
    """
    Rows of vals correspond to models
    Columns of vals correspond to metrics
    """
    if type(vals) is not pd.DataFrame:
        eval_df = pd.DataFrame(vals, index=model_names, columns=['mse', 'asym'])
    else:
        eval_df = vals

    eval_df.plot.bar(**bar_param)

def vis_by_scatter(y_true_list, y_pred_list, model_name_list, vals, grid_size, margins=(0.5,0.5), fig_size=(14,8)):
    _, axes = plt.subplots(*grid_size)
    plt.subplots_adjust(wspace=margins[0], hspace=margins[1])

    n_prev_plots = 0
    for k, model_names in enumerate(model_name_list):
        y_true = y_true_list[k]
        n_instances = y_true_list[k].shape[0]
        for i, model_name in enumerate(model_names):
            y_pred = y_pred_list[k][:, i]

            lim = np.min(y_true), np.max(y_true)

            # perfect line
            ax = axes.flatten()[n_prev_plots + i]
            ax.plot(lim, lim, '-', color='#444444', linewidth=1, label='perfect')
            ax.margins(x=0.05, y=0.05, tight=False)

            # error line
            guide_mses = [100, 500]
            guide_mse_styles = ['--', ':']
            for g, ls in zip(guide_mses, guide_mse_styles):
                ax.plot(lim, lim + np.sqrt(g)/n_instances, linestyle=ls, color='#444444', linewidth=1, label='MSE {}'.format(g))
                ax.plot(lim, lim - np.sqrt(g)/n_instances, linestyle=ls, color='#444444', linewidth=1)

            # true vs. pred
            vs_df = pd.DataFrame(np.array([y_true,y_pred]).T, columns=['Truth', 'Predicted'])
            vs_df.plot.scatter(x='Truth', y='Predicted', ax=ax, figsize=fig_size, c='#FF5555', title=model_name)

#             ax.legend(prop={'size':12})
            if vals is not None:
                text_list = [metric + ': {:.2f}'.format(vals[k].iloc[i][metric]) for metric in vals[k].columns.values]
                xpos = ax.get_xlim()[0] + 0.05*(ax.get_xlim()[1] - ax.get_xlim()[0])
                ypos = ax.get_ylim()[0] + 0.95*(ax.get_ylim()[1] - ax.get_ylim()[0])
                ax.text(xpos, ypos, s='\n'.join(text_list),
                       ha='left', va='top')

        n_prev_plots += len(model_names)
