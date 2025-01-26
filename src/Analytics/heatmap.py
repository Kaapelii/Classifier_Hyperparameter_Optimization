import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import config

def make_heatmap(gridsearch, time_spent, is_successive_halving=False, name="Optimization Heatmap"):
    """Helper to make a heatmap."""
    results = pd.DataFrame(gridsearch.cv_results_)
    
    # Adjust the parameter names to match the actual parameter names used in the pipeline
    param_C_col = [col for col in results.columns if 'classifier__C' in col][0]
    param_gamma_col = [col for col in results.columns if 'classifier__gamma' in col][0]
    
    # Filter out non-numeric values before converting to float
    numeric_results = results[results[param_gamma_col].apply(lambda x: isinstance(x, (int, float)))]
    numeric_results = numeric_results[numeric_results[param_C_col].apply(lambda x: isinstance(x, (int, float)))]

    numeric_results[[param_C_col, param_gamma_col]] = numeric_results[[param_C_col, param_gamma_col]].astype(np.float64)
    
    if is_successive_halving:
        # SH dataframe: get mean_test_score values for the highest iter
        scores_matrix = numeric_results.sort_values("iter").pivot_table(
            index=param_gamma_col,
            columns=param_C_col,
            values="mean_test_score",
            aggfunc="last",
        )
    else:
        scores_matrix = numeric_results.pivot_table(
            index=param_gamma_col, columns=param_C_col, values="mean_test_score", aggfunc="mean"
        )

    # Fill missing values with np.nan
    scores_matrix = scores_matrix.fillna(np.nan)

    fig, ax = plt.subplots(figsize=(config.HEATMAP_W, config.HEATMAP_H))
    im = ax.imshow(scores_matrix, aspect='auto', cmap='viridis')

    param_C_values = numeric_results[param_C_col].unique()
    param_gamma_values = numeric_results[param_gamma_col].unique()

    ax.set_xticks(np.arange(len(param_C_values)))
    ax.set_xticklabels(["{:.0E}".format(x) for x in param_C_values])
    ax.set_xlabel("C", fontsize=10)

    ax.set_yticks(np.arange(len(param_gamma_values)))
    ax.set_yticklabels(["{:.0E}".format(x) for x in param_gamma_values])
    ax.set_ylabel("gamma", fontsize=10)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)

    if is_successive_halving:
        iterations = numeric_results.pivot_table(
            index=param_gamma_col, columns=param_C_col, values="iter", aggfunc="max"
        ).values
        for i in range(len(param_gamma_values)):
            for j in range(len(param_C_values)):
                ax.text(
                    j,
                    i,
                    iterations[i, j],
                    ha="center",
                    va="center",
                    color="w",
                    fontsize=10,
                )

    fig.subplots_adjust(right=(config.HEATMAP_W*0.14), bottom=config.HEATMAP_H*0.02) 
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    cbar_ax.set_ylabel("mean_test_score", rotation=-90, va="bottom", fontsize=10)

    ax.set_title(f"{name}\ntime = {time_spent:.3f}s", fontsize=12)
    plt.savefig(os.path.join(config.HEATMAP_DIR, 'Heatmap_' + name + '.png'))
    plt.close(fig)