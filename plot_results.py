import matplotlib.pyplot as plt
import os


def plot_results(df):
    # Define colors for each base algorithm
    colors = {
           'SVD': 'blue',
            'NMF': 'red',
            'KNN': 'purple',
            'Base': 'orange',
            'Bayesian-BMF': 'green',
            'GPT4oMini': 'teal',
            'GPT4Turbo': 'magenta',
            'Mean' : 'Black'
    }

    # Define line styles for each type
    line_styles = {
        'Default': 'solid',
        'Bagging': 'dashed',
        'BaggingV2': 'dotted'
    }

    # Extract the base model name and type from the 'model' column and remove 'Algo'
    df['base_model'] = df['model'].apply(lambda x: x.split(' ')[0].replace('Algo', ''))
    df['type'] = df['model'].apply(lambda x: x.split('(')[-1].split(')')[0].replace('Algo', ''))
    oracle_name = df['oracle'][0]

    # create the output repository
    if not os.path.exists(f"curves/{oracle_name}"):
        os.makedirs(f"curves/{oracle_name}")

    # Determine if 'rank' or 'leaf_probability' is available
    x_column = 'rank' if 'rank' in df.columns else 'leaf_probability'

    # Plot RMSE vs Rank or Leaf Probability (depending on availability)
    if x_column in df.columns and not df[x_column].isnull().all() and len(df[x_column].unique()) > 1:
        plt.figure()
        for algo in df['base_model'].unique():
            for t in line_styles:
                subset = df[(df['base_model'] == algo) & (df['type'] == t)]
                if not subset.empty:
                    label = f"{algo} {t}" if t != "Default" else algo
                    plt.plot(subset[x_column], subset['rmse-error'].apply(lambda x: float(x.split(' - ')[0])),
                             label=label, color=colors[algo], linestyle=line_styles[t])
        plt.xlabel("Rank" if x_column == 'rank' else "Leaf Probability")
        plt.ylabel('RMSE')
        #plt.title(f"RMSE vs {x_column.capitalize()} , Noise = {df['noise_level'].iloc[0]}")
        plt.title(f"RMSE vs {x_column.capitalize()}")
        plt.legend()
        plt.legend().set_visible(False)
        plt.savefig(f'curves/{oracle_name}/rmse_vs_{x_column}.pdf')


    # Plot RMSE vs Training Ratio
    if 'training_ratio' in df.columns and len(df['training_ratio'].unique()) > 1:
        plt.figure()
        for algo in df['base_model'].unique():
            for t in line_styles:
                subset = df[(df['base_model'] == algo) & (df['type'] == t)]
                if not subset.empty:
                    label = f"{algo} {t}" if t != "Default" else algo
                    plt.plot(subset['training_ratio'], subset['rmse-error'].apply(lambda x: float(x.split(' - ')[0])),
                             label=label, color=colors[algo], linestyle=line_styles[t])
        plt.xlabel('Training Ratio')
        plt.ylabel('RMSE')
        plt.title(f"RMSE vs Training Ratio")
        plt.title(f"RMSE vs Training Ratio , Noise = {df['noise_level'].iloc[0]}")
        plt.legend()
        plt.legend().set_visible(False)
        plt.savefig(f'curves/{oracle_name}/rmse_vs_training_ratio.pdf')


    # Plot RMSE vs Noise Level
    if 'noise_level' in df.columns and len(df['noise_level'].unique()) > 1:
        plt.figure()
        for algo in df['base_model'].unique():
            for t in line_styles:
                subset = df[(df['base_model'] == algo) & (df['type'] == t)]
                if not subset.empty:
                    label = f"{algo} {t}" if t != "Default" else algo
                    plt.plot(subset['noise_level'], subset['rmse-error'].apply(lambda x: float(x.split(' - ')[0])),
                             label=label, color=colors[algo], linestyle=line_styles[t])
        plt.xlabel('Noise Level')
        plt.ylabel('RMSE')
        plt.title('RMSE vs Noise Level')
        plt.legend()
        plt.legend().set_visible(False)
        plt.savefig(f'curves/{oracle_name}/rmse_vs_noise_level.pdf')

    plt.show()
    plt.close()
