import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
from PIL import Image
import glob

# Function to plot multiple Y axes on a single chart
def plot_multi_y_axis(data: pd.DataFrame, x_col: str, y_cols: List[str], title: str = '',sameaxis=False) -> None:
    """
    Plots multiple Y-axes on a single chart with given DataFrame.
    
    Args:
    data: DataFrame containing the data.
    x_col: The name of the column to use for the X-axis.
    y_cols: A list of column names for multiple Y-axes.
    title: Title of the plot.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=200)
    colors = [
        'blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 
        'cyan', 'magenta', 'yellow', 'black', 'lightblue', 'lightgreen', 
        'darkred', 'gold', 'lime', 'lavender', 'turquoise', 'darkblue',
        'olive', 'maroon', 'aquamarine', 'navy', 'teal', 'coral', 'crimson', 
        'orchid', 'salmon', 'khaki'
    ]
    markers = [
        'o', 's', '^', 'd', '>', '<', 'p', '*', 'h', 'H', '+', 'x', 
        'D', '|', '_', 'X', '1', '2', '3', '4', '8', 'P', 'v', 
        'A', 'B', 'C', 'V', 'M', 'W', 'T', 'd', 'o', 'h'
    ]
    sns.scatterplot(x=x_col, y=y_cols[0], data=data, ax=ax1, color=colors[0], s=50, marker=markers[0],label=y_cols[0])
    sns.lineplot(x=x_col, y=y_cols[0], data=data, ax=ax1, color=colors[0], linestyle='--', alpha=0.7)
    ax1.set_ylabel(y_cols[0], color=colors[0])
    ax1.tick_params(axis='y', labelcolor=colors[0])
    ax1.legend()
    ax1.get_legend().remove()
    axes = [ax1]
    for i, col in enumerate(y_cols[1:], 1):
        if sameaxis:
            ax=ax1
        else:
            ax = ax1.twinx()
        position = ('axes', 1 + 0.1 * (i//2)) if i % 2 == 1 else ('axes', -0.1 * (i//2))
        ax.spines['right' if i % 2 == 1 else 'left'].set_position(position)
        if i % 2 == 0:
            ax.yaxis.set_label_position('left')
            ax.yaxis.tick_left()
        sns.scatterplot(x=x_col, y=col, data=data, ax=ax, color=colors[i % len(colors)], s=50, marker=markers[i % len(markers)],label=col)
        sns.lineplot(x=x_col, y=col, data=data, ax=ax, color=colors[i % len(colors)], linestyle='--', alpha=0.7)
        ax.set_ylabel(col, color=colors[i % len(colors)])
        ax.tick_params(axis='y', labelcolor=colors[i % len(colors)])
        ax.legend()
        ax.get_legend().remove()
        axes.append(ax)
    plt.title(title)
    ax1.set_xticklabels([f"{idx}: {system}" for idx, system in zip(data.index, data[x_col])], rotation=45)
    ax1.grid()
    if sameaxis:
        ax.set_ylabel('')
        ax.tick_params(axis='y', labelcolor='black')
        plt.legend()
    plt.show()

# Function to display images with annotations
def display_images(data: pd.DataFrame, patterns: Dict[str, str], annotations: List[tuple], dpi: int = 200,title='') -> None:
    """
    Displays images with annotations from DataFrame columns specified by patterns.
    
    Args:
    data: DataFrame containing the data.
    patterns: Dictionary with keys as DataFrame column names and values as glob patterns for image paths.
    annotations: List of tuples with annotation settings. Its str would be formatted with the data with a special key "name" for the index.
    dpi: Resolution for the images displayed.
    title: title pattern for the images. This would be formatted with the data with a special key "name" for the index.
    eg:
    1. Show figures of occupations:
    
        patterns = {
        'conduction_occup_image_path': '*heatmap_Emin1.420eV_Emax1.610eV_fitted*',
        'valence_occup_image_path': '*_heatmap_Emin-0.131eV_Emax0.010eV*',
        }

        annotations = [
            (0, "mu_Boltz(eV)={mu_Boltz(eV):.3f}\nT_Boltz(K)={T_Boltz(K):.1f}\nFinal_Max_Change={Max_Occupation_change_Conduction:.3f}", 
            ((0.8, 0.8), 'right', 'top')),  
            (1, "Final_Max_Change={Max_Occupation_change_Valence:.3f}", 
            ((0.2, 0.2), 'left', 'bottom'))
        ]
        title="No.{name} {System}"
        display_images(gaas_data, patterns, annotations,title)
    
    2. Show figures of current:
    
        current_annotations = []
        directions = ['x', 'y', 'z']
        components = ['', '_diag', '_offdiag']
        for dir_i, dir in enumerate(directions):
            for compo_i, compo in enumerate(components):
                annotation_text = f"Final_DC{compo}_{dir}={{DC{compo}_{dir}:.2e}}"
                position = ((compo_i * 2 + 1) / 6, (dir_i * 2 + 1) / 6)  
                current_annotations.append((0, annotation_text, (position, 'center', 'center')))  

        current_patterns = {'current_image_path': '*j_smooth_off*'}
        title="No.{name} {System}"
        display_images(gaas_data, current_patterns, current_annotations, title)
    

    """
    for col, pattern in patterns.items():
        data[col] = data.index.to_series().apply(lambda x: glob_image_path(x, pattern))
    first_image_path = data.iloc[0][list(patterns.keys())[0]]
    image = Image.open(first_image_path)
    width, height = image.size
    fig, axs = plt.subplots(len(patterns), len(data), figsize=(width / dpi * len(data), height / dpi * len(patterns)), dpi=dpi)
    for ith, (idx, row) in enumerate(data.iterrows()):
        for i, img_col in enumerate(patterns.keys()):
            ax = axs[i, ith] if len(patterns) > 1 else axs[ith]
            if i==0:
                ax.set_title(title.format(name=idx,**row))
            image_path = row[img_col]
            image = Image.open(image_path)
            ax.imshow(image)
            ax.axis('off')
            for ann in annotations:
                if ann[0] == i:
                    ax.annotate(ann[1].format(name=idx,**row),
                                xy=ann[2][0],
                                xycoords='axes fraction',
                                horizontalalignment=ann[2][1],
                                verticalalignment=ann[2][2],
                                fontsize=ann[3] if len(ann) > 3 else 10)
    plt.tight_layout()
    plt.show()

def glob_image_path(directory: str, pattern: str) -> str:
    """
    Returns the first file path matching the pattern in the specified directory.
    
    Args:
    directory: The directory to search.
    pattern: The glob pattern to match files.
    
    Returns:
    The first matched file path or None if no file is found.
    """
    files = glob.glob(f"{directory}/{pattern}")
    return files[0] if files else None
def plot_correlation_heatmap(data: pd.DataFrame, columns: List[str], title: str, cmap: str = 'coolwarm') -> None:
    """
    Plots a heatmap of the correlation matrix for the selected numerical columns in the DataFrame.

    Args:
    data (pd.DataFrame): The DataFrame containing the data.
    columns (List[str]): A list of column names to include in the correlation matrix. Make sure these column are numerical type.
    title (str): The title for the heatmap.
    cmap (str): The colormap for the heatmap. Default is 'coolwarm'.
    """
    # Ensure the figure size is large enough to clearly show the heatmap
    plt.figure(figsize=(15, 10))
    
    # Select the numeric columns and calculate the correlation matrix
    corr_matrix = data[columns].corr()
    
    # Create the heatmap using seaborn
    sns.heatmap(corr_matrix, annot=True, fmt=".1f", cmap=cmap, cbar=True, vmin=-1, vmax=1)
    
    # Set the title for the heatmap
    plt.title(title)
    
    # Display the plot
    plt.show()