import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- CONFIG ---
MODEL_MAPPING = {
    'efficientnet-base': 'EfficientNet',
    'gigapath-base': 'GigaPath',
    'resnet-base': 'ResNet',
    'swin-base': 'Swin Transformer',
    'uni-base': 'UNI',
    'virchow2-base': 'Virchow2',
    'vit-base': 'ViT'
}

def main(input_csv, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 1. Data Preparation
    if not os.path.exists(input_csv):
        print(f"Error: File {input_csv} does not exist.")
        return

    # Load the new table with bootstrap results
    df = pd.read_csv(input_csv)
    df_global = df[df['Dataset'] == 'GLOBAL'].copy()

    # Map model names (according to the new -base convention)
    df_global['Model'] = df_global['Model'].map(MODEL_MAPPING).fillna(df_global['Model'])

    # Calculate Semantic Shift from averaged Cosine Similarity
    df_global['Semantic_Shift'] = 1 - df_global['CosSim_Mean']
    
    # Softmax Delta (Confidence Drop) is already available in the table
    df_global['Confidence_Drop'] = df_global['Softmax_Delta']

    # 2. Plot Creation
    plt.figure(figsize=(13, 10))
    sns.set_style("whitegrid", {'axes.grid': True, 'grid.linestyle': '--'})

    # Scatter plot: Color = Artifact, Shape = Architecture
    scatter = sns.scatterplot(
        data=df_global,
        x='Semantic_Shift',
        y='Confidence_Drop',
        hue='Artifact',
        style='Model',
        s=250, 
        alpha=0.85, 
        edgecolor='k',
        palette='Set1'
    )

    # TREND LINE with confidence interval
    # The shaded area around the trend line effectively represents the significance of the correlation
    sns.regplot(
        data=df_global, 
        x='Semantic_Shift', 
        y='Confidence_Drop', 
        scatter=False, 
        color='black', 
        line_kws={"linestyle": "--", "alpha": 0.4, "label": "Global Trend Line"}
    )

    # --- DANGER ZONE (RED ZONE / OVERCONFIDENCE) ---
    # Models in this area have high feature drift (they see something is wrong), 
    # but their confidence (softmax) barely drops. This is clinically dangerous.
    x_min_danger = 0.15
    y_max_danger = 0.05
    y_min_plot = df_global['Confidence_Drop'].min() - 0.03

    plt.fill_between([x_min_danger, df_global['Semantic_Shift'].max() * 1.2], 
                     y_min_plot, y_max_danger, 
                     color='red', alpha=0.07, label='High Risk Zone (Overconfident)')

    plt.axvline(x=x_min_danger, color='red', alpha=0.3, linestyle=':')
    plt.axhline(y=0, color='black', alpha=0.3, linewidth=1)

    # Aesthetics
    plt.title("Global Overconfidence Analysis: Latent Drift vs. Confidence Loss", 
              fontsize=18, fontweight='bold', pad=20)
    plt.xlabel("Semantic Shift (1 - Mean Cosine Similarity)", fontsize=13, fontweight='bold')
    plt.ylabel(r"Confidence Drop (Softmax $\Delta$)", fontsize=13, fontweight='bold')

    # Legend configuration - pulled outside so it doesn't overlap points
    plt.legend(
        loc='upper left', 
        bbox_to_anchor=(1.02, 1), 
        fontsize=10, 
        frameon=True, 
        shadow=True, 
        ncol=1, 
        labelspacing=1.1, 
        borderpad=1
    )

    # Axis limits
    plt.xlim(0, df_global['Semantic_Shift'].max() * 1.1)
    plt.ylim(y_min_plot, df_global['Confidence_Drop'].max() * 1.1)

    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join(output_dir, "overconfidence_analysis_scatter.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Overconfidence plot generated successfully in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate overconfidence scatter plot.")
    parser.add_argument("--input", type=str, required=True, help="Path to the evaluated CSV results.")
    parser.add_argument("--output-dir", type=str, default="./results/charts", help="Directory to save the charts.")
    
    args = parser.parse_args()
    main(args.input, args.output_dir)