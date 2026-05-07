import os
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIG ---
# Definition of mapping and strictly defined order (consistent with radar and tornado)
MODEL_MAPPING = {
    'resnet-base': 'ResNet',
    'efficientnet-base': 'EfficientNet',
    'vit-base': 'ViT',
    'swin-base': 'Swin Transformer',
    'gigapath-base': 'GigaPath',
    'uni-base': 'UNI',
    'virchow2-base': 'Virchow2'
}

def generate_final_latent_panel(df, dataset_name, ordered_model_names, output_dir):
    subset = df[df['Dataset'] == dataset_name.upper()].copy()
    if subset.empty: 
        print(f"No data found for dataset: {dataset_name}")
        return

    sns.set_style("white")
    # Increase height slightly for better label readability
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 9))
    
    # --- PANEL 1: COSINE DISTANCE (SEMANTIC DRIFT) ---
    pivot_cos = subset.pivot(index='Model', columns='Artifact', values='Cosine_Distance')
    pivot_cos = pivot_cos.reindex(ordered_model_names)
    
    # Draw heatmap - the redder, the greater the semantic 'drift'
    sns.heatmap(pivot_cos, annot=True, cmap="YlOrRd", fmt=".3f", 
                ax=ax1, annot_kws={"size": 12, "weight": "bold"}, 
                cbar_kws={'label': 'Semantic Drift (1 - CosSim)'})
    
    ax1.set_title(f"A: Latent Drift (Cosine Distance) - {dataset_name}", 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel("Architecture", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Artifact", fontsize=13, fontweight='bold')

    # --- PANEL 2: NORMALIZED EUCLIDEAN DISTANCE (PERTURBATION) ---
    pivot_l2 = subset.pivot(index='Model', columns='Artifact', values='L2_Norm')
    pivot_l2 = pivot_l2.reindex(ordered_model_names)
    
    sns.heatmap(pivot_l2, annot=True, cmap="YlOrRd", fmt=".2f", 
                ax=ax2, annot_kws={"size": 12, "weight": "bold"}, 
                cbar_kws={'label': 'Relative L2 Perturbation'})
    
    ax2.set_title(f"B: Feature Perturbation (Norm. L2) - {dataset_name}", 
                  fontsize=16, fontweight='bold', pad=20)
    
    # Axis aesthetics
    ax1.tick_params(labelsize=11)
    ax2.tick_params(labelsize=11)
    ax2.set_ylabel("") # Remove Y-axis label for the second panel as it's identical
    ax2.set_xlabel("Artifact", fontsize=13, fontweight='bold')

    # Common title for the whole panel
    plt.suptitle(f"LATENT SPACE STABILITY PROFILE: {dataset_name.upper()}", 
                 fontsize=22, y=1.02, fontweight='bold')
    
    plt.tight_layout()
    
    filename = os.path.join(output_dir, f"latent_profile_{dataset_name.lower()}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved latent profile: {filename}")
    plt.close()

def main(input_csv, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 1. Data Preparation 
    if not os.path.exists(input_csv):
        print(f"Error: File {input_csv} does not exist.")
        return

    df = pd.read_csv(input_csv)

    # Cleaning and mapping
    df['Model'] = df['Model'].map(MODEL_MAPPING).fillna(df['Model'])

    # Calculate Cosine Distance (1 - CosSim_Mean)
    df['Cosine_Distance'] = 1 - df['CosSim_Mean']

    # Normalize Euclidean Distance per model (for better visibility of relative impact)
    # Using EucDist_Mean from bootstrap
    df['L2_Norm'] = df.groupby('Model')['EucDist_Mean'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
    )

    # Force row order in the heatmap based on the mapping definition
    ordered_model_names = [m for m in MODEL_MAPPING.values() if m in df['Model'].unique()]

    # Generate profiles for all views
    for ds in ['GLOBAL', 'TCGA', 'ANORAK']:
        generate_final_latent_panel(df, ds, ordered_model_names, output_dir)
        
    print(f"All latent profile charts generated successfully in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate latent space stability heatmaps.")
    parser.add_argument("--input", type=str, required=True, help="Path to the evaluated CSV results.")
    parser.add_argument("--output-dir", type=str, default="./results/charts", help="Directory to save the charts.")
    
    args = parser.parse_args()
    main(args.input, args.output_dir)