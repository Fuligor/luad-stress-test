import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import os
import argparse

# --- CONFIG ---
MODEL_MAPPING = {
    'resnet-base': 'ResNet', 
    'efficientnet-base': 'EfficientNet',
    'vit-base': 'ViT', 
    'swin-base': 'Swin',
    'gigapath-base': 'GigaPath', 
    'uni-base': 'UNI', 
    'virchow2-base': 'Virchow2'
}

# Mapping artifact names (adjust to the names in your results folder)
# If the CSV has 'stitching' instead of 'stitch', this mapping will fix it
ART_MAP = {
    'blur': 'blur', 'stitching': 'stitch', 'marker': 'marker',
    'dust': 'dust', 'precipitate': 'prec', 'fold': 'fold'
}

TARGET_ARTIFACTS = ['blur', 'stitch', 'marker', 'dust', 'prec', 'fold']
ORDERED_MODELS = ['ResNet', 'EfficientNet', 'ViT', 'Swin', 'GigaPath', 'UNI', 'Virchow2']

def main(input_csv, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load and prepare data
    if not os.path.exists(input_csv):
        print(f"Error: File {input_csv} does not exist.")
        return
        
    df = pd.read_csv(input_csv)
    
    # Map models (according to the -base convention)
    df['Model'] = df['Model'].map(MODEL_MAPPING).fillna(df['Model'])
    df['Artifact_Clean'] = df['Artifact'].map(ART_MAP).fillna(df['Artifact'])
    
    df_plot = df[df['Dataset'].isin(['TCGA', 'ANORAK'])].copy()
    models = [m for m in ORDERED_MODELS if m in df_plot['Model'].unique()]
    
    # 2. Grid configuration
    fig, axes = plt.subplots(2, 3, figsize=(20, 12), sharex=False, sharey=True)
    axes = axes.flatten()
    
    # Colors for Cohorts (TCGA vs ANORAK)
    palette = {'TCGA': '#1f77b4', 'ANORAK': '#ff7f0e'}
    # Shapes for Models
    model_markers = ['o', 's', 'D', '^', 'v', 'X', 'P'] 
    markers_map = {model: model_markers[i] for i, model in enumerate(models)}
    
    for idx, artifact in enumerate(TARGET_ARTIFACTS):
        ax = axes[idx]
        art_df = df_plot[df_plot['Artifact_Clean'] == artifact]
        
        # --- DRAWING CONNECTING LINES (Transitions) ---
        for model in models:
            m_data = art_df[art_df['Model'] == model]
            if len(m_data) >= 2:
                tcga_pt = m_data[m_data['Dataset'] == 'TCGA']
                anorak_pt = m_data[m_data['Dataset'] == 'ANORAK']
                
                if not tcga_pt.empty and not anorak_pt.empty:
                    # Draw a line between the TCGA point and the ANORAK point
                    ax.plot(
                        [tcga_pt['Base_F1'].values[0], anorak_pt['Base_F1'].values[0]],
                        [tcga_pt['RPD'].values[0], anorak_pt['RPD'].values[0]],
                        color='gray', linestyle='-', linewidth=1.2, alpha=0.3, zorder=1
                    )
        
        # --- DRAWING POINTS ---
        for ds in ['TCGA', 'ANORAK']:
            ds_df = art_df[art_df['Dataset'] == ds]
            for model in models:
                m_pt = ds_df[ds_df['Model'] == model]
                if not m_pt.empty:
                    ax.scatter(
                        m_pt['Base_F1'], m_pt['RPD'], 
                        color=palette[ds], marker=markers_map[model], 
                        s=150, edgecolor='black', linewidth=0.8, zorder=2, alpha=0.9
                    )

        # --- ARTIFACT TITLE ---
        ax.set_title(artifact.upper(), fontsize=16, fontweight='bold', pad=10)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        if idx >= 3: 
            ax.set_xlabel("Baseline F1 Score", fontsize=13, fontweight='bold')
        if idx % 3 == 0: 
            ax.set_ylabel("RPD (%)", fontsize=13, fontweight='bold')

    # 3. LEGEND
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=palette['TCGA'], 
               markeredgecolor='black', markersize=12, label='TCGA Cohort'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=palette['ANORAK'], 
               markeredgecolor='black', markersize=12, label='ANORAK Cohort'),
        Line2D([0], [0], color='none', label='  '), # Spacer
    ]
    for model in models:
        legend_elements.append(Line2D([0], [0], marker=markers_map[model], color='w', 
                                      markerfacecolor='gray', markeredgecolor='black', 
                                      markersize=10, label=model))

    plt.tight_layout(rect=[0, 0.08, 1, 0.95]) 
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02), 
               ncol=5, frameon=True, fontsize=12)
    
    plt.suptitle("COHORT TRANSITION ANALYSIS: STABILITY ACROSS DATASETS", 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Save the plot
    save_path = os.path.join(output_dir, "cohort_transitions_final.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Cohort transition plots generated successfully in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate cohort transition plots.")
    parser.add_argument("--input", type=str, required=True, help="Path to the evaluated CSV results.")
    parser.add_argument("--output-dir", type=str, default="./results/charts", help="Directory to save the charts.")
    
    args = parser.parse_args()
    main(args.input, args.output_dir)