import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse

def main(input_file, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Data
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} does not exist.")
        return

    df = pd.read_csv(input_file)
    
    # Filter only GLOBAL (all models combined)
    df_global = df[df['Dataset'] == 'GLOBAL'].copy()

    # Group by artifacts to show average impact across all models
    # Aggregate bootstrap means and their standard deviations
    tornado_data = df_global.groupby('Artifact').agg({
        'PFR_Mean': 'mean',
        'PFR_Std': 'mean',  # Average standard deviation between models
        'NFR_Mean': 'mean',
        'NFR_Std': 'mean'
    }).reset_index()

    tornado_data['Artifact'] = tornado_data['Artifact'].str.upper()

    # Sort by total impact (PFR + NFR)
    tornado_data['total_impact'] = tornado_data['PFR_Mean'] + tornado_data['NFR_Mean']
    tornado_data = tornado_data.sort_values('total_impact', ascending=True)

    # 2. Plot Initialization (Size synchronized with Radar: 8 x 8.5)
    fig, ax = plt.subplots(figsize=(8, 8.5))
    
    artifacts = tornado_data['Artifact'].tolist()
    y_pos = np.arange(len(artifacts))

    # PFR Bars (Critical Risk) - Right side
    ax.barh(y_pos, tornado_data['PFR_Mean'], 
            xerr=tornado_data['PFR_Std'], # ADDED ERROR BARS
            error_kw={'ecolor': '#a50026', 'linewidth': 1.5, 'capsize': 3},
            color='#d73027', alpha=0.85, 
            label='Critical Risk (Cancer $\\rightarrow$ Normal)')

    # NFR Bars (False Alarms) - Left side
    # Use negative values for left orientation, but error bars must be positive
    ax.barh(y_pos, -tornado_data['NFR_Mean'], 
            xerr=tornado_data['NFR_Std'], # ADDED ERROR BARS
            error_kw={'ecolor': '#313695', 'linewidth': 1.5, 'capsize': 3},
            color='#4575b4', alpha=0.85, 
            label='False Alarms (Normal $\\rightarrow$ Cancer)')

    # 3. Add text labels (at the ends of the bars)
    for i, (pfr, nfr) in enumerate(zip(tornado_data['PFR_Mean'], 
                                       tornado_data['NFR_Mean'])):
        # PFR value (right)
        ax.text(pfr + 2, i, f"{pfr:.1f}%", 
                va='center', ha='left', fontsize=11, fontweight='bold', color='#a50026')
        
        # NFR value (left)
        ax.text(-nfr - 2, i, f"{nfr:.1f}%", 
                va='center', ha='right', fontsize=11, fontweight='bold', color='#313695')

    # 4. Aesthetics and Formatting
    ax.axvline(0, color='black', linewidth=1.5, zorder=3)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(artifacts, fontsize=12, fontweight='bold')
    
    # Set X-axis limits so labels and error bars fit
    max_val = max(tornado_data['PFR_Mean'].max() + tornado_data['PFR_Std'].max(), 
                  tornado_data['NFR_Mean'].max() + tornado_data['NFR_Std'].max())
    max_limit = max_val + 15
    ax.set_xlim(-max_limit, max_limit)
    
    # Remove minus signs from X-axis ticks
    ticks = ax.get_xticks()
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{abs(t):.0f}%" for t in ticks], fontsize=11)
    
    ax.set_xlabel('Clinical Impact Score (%)', fontsize=14, fontweight='bold', labelpad=15)
    
    # Title consistent with Radar
    ax.set_title('Clinical Vulnerability Tornado\n(Aggregate Impact on LUAD Classification)', 
                 fontsize=15, fontweight='bold', pad=15)

    # Legend below the plot
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), 
              ncol=1, frameon=False, fontsize=11) 

    ax.grid(axis='x', linestyle='--', alpha=0.4, zorder=0)
    sns.despine(left=True, bottom=True, top=True, right=True)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    # 5. Save Files
    base_save_path = os.path.join(output_dir, "clinical_impact_tornado_bootstrap")
    plt.savefig(f"{base_save_path}.png", dpi=300)
    plt.savefig(f"{base_save_path}.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"Tornado plots generated successfully in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate clinical vulnerability tornado plots.")
    parser.add_argument("--input", type=str, required=True, help="Path to the evaluated CSV results.")
    parser.add_argument("--output-dir", type=str, default="./results/charts", help="Directory to save the charts.")
    
    args = parser.parse_args()
    main(args.input, args.output_dir)