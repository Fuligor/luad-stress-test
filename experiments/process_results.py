import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

# --- CONFIG ---
MODELS = ["efficientnet-base", "gigapath-base", "resnet-base", "swin-base", "uni-base", "virchow2-base", "vit-base"]
ALL_CLASSES = ['NTU', 'ACC', 'CRB', 'LEP', 'MIP', 'PAP', 'SOL']
CANCER_CLASSES = ['ACC', 'CRB', 'LEP', 'MIP', 'PAP', 'SOL']
NORMAL_CLASS = 'NTU'
N_BOOTSTRAP = 500
RANDOM_SEED = 42 

def clean_filename(s):
    return os.path.splitext(str(s).strip())[0]

def calculate_ece_numpy(confidences, accuracies, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if np.any(mask):
            ece += np.abs(accuracies[mask].mean() - confidences[mask].mean()) * (mask.sum() / len(confidences))
    return ece

def run_consistent_bootstrap(data_dict, n_iterations=N_BOOTSTRAP):
    res = []
    n = len(data_dict['y_true'])
    is_cancer = np.isin(data_dict['y_true'], CANCER_CLASSES)
    is_normal = (data_dict['y_true'] == NORMAL_CLASS)
    
    y_t = data_dict['y_true']
    y_b = data_dict['y_pred_b']
    y_a = data_dict['y_pred_a']
    c_b = data_dict['conf_b']
    c_a = data_dict['conf_a']
    v_b = data_dict['vec_b'].astype(np.float32)
    v_a = data_dict['vec_a'].astype(np.float32)
    
    # Fix the same data for consistency
    np.random.seed(RANDOM_SEED)
    
    for _ in range(n_iterations):
        idx = np.random.choice(n, n, replace=True)
        
        # 1. F1 Scores
        f_b = f1_score(y_t[idx], y_b[idx], average='macro', labels=ALL_CLASSES, zero_division=0)
        f_a = f1_score(y_t[idx], y_a[idx], average='macro', labels=ALL_CLASSES, zero_division=0)
        
        # 2. Precision/Recall
        p_a = precision_score(y_t[idx], y_a[idx], average='macro', labels=ALL_CLASSES, zero_division=0)
        r_a = recall_score(y_t[idx], y_a[idx], average='macro', labels=ALL_CLASSES, zero_division=0)
        
        # 3. Robustness Score
        rpd = ((f_b - f_a) / (f_b + 1e-9)) * 100
        rob_r = (f_a / (f_b + 1e-9)) * 100
        
        # 4. Calibration & Confidence (ECE & Softmax Delta)
        ece_b = calculate_ece_numpy(c_b[idx], (y_b[idx] == y_t[idx]))
        ece_a = calculate_ece_numpy(c_a[idx], (y_a[idx] == y_t[idx]))
        ece_d = ece_a - ece_b
        sof_d = np.mean(c_b[idx] - c_a[idx]) 
        
        # 5. Flips (PFR/NFR)
        correct_b = (y_b[idx] == y_t[idx])
        pfr_mask = correct_b & is_cancer[idx]
        pfr = (y_a[idx][pfr_mask] == NORMAL_CLASS).sum() / pfr_mask.sum() * 100 if pfr_mask.sum() > 0 else 0
        
        nfr_mask = correct_b & is_normal[idx]
        nfr = np.isin(y_a[idx][nfr_mask], CANCER_CLASSES).sum() / nfr_mask.sum() * 100 if nfr_mask.sum() > 0 else 0

        # 6. Embedding Stability (Cosine similarity & Euclidean distance)
        dot = np.sum(v_b[idx] * v_a[idx], axis=1)
        norms = np.linalg.norm(v_b[idx], axis=1) * np.linalg.norm(v_a[idx], axis=1)
        cos = dot / (norms + 1e-9)
        
        euc = np.linalg.norm(v_b[idx] - v_a[idx], axis=1) / np.sqrt(v_b.shape[1])
        
        res.append([f_b, f_a, p_a, r_a, rpd, rob_r, ece_d, pfr, nfr, np.mean(cos), np.mean(euc), sof_d])

    arr = np.array(res)
    cols = ['f1_b', 'f1_a', 'p_a', 'r_a', 'rpd', 'rob_r', 'ece_d', 'pfr', 'nfr', 'cos', 'euc', 'sof_d']
    return {name: (np.mean(arr[:, i]), np.std(arr[:, i]), np.percentile(arr[:, i], 2.5), np.percentile(arr[:, i], 97.5)) for i, name in enumerate(cols)}

def main(base_path, output_path):
    full_results = []

    for model_name in tqdm(MODELS, desc="Models Progress"):
        model_dir = os.path.join(base_path, model_name)
        if not os.path.exists(model_dir): 
            continue
        
        data_p, data_e = {}, {}
        subdirs = [d for d in os.listdir(model_dir) if "-" in d and "dhmc" not in d.lower()]
        
        for sub in subdirs:
            try:
                ds, art = sub.split("-")
                csv_path = os.path.join(model_dir, sub, "predictions.csv")
                emb_path = os.path.join(model_dir, sub, "embedings.parquet")
                if os.path.exists(csv_path) and os.path.exists(emb_path):
                    # Predictions
                    df_p = pd.read_csv(csv_path)
                    df_p.rename(columns={df_p.columns[0]: 'filename'}, inplace=True)
                    df_p['filename'] = df_p['filename'].apply(clean_filename)
                    data_p[(ds, art)] = df_p
                    
                    # Embeddings
                    df_e = pd.read_parquet(emb_path)
                    if not isinstance(df_e.index, pd.RangeIndex): df_e = df_e.reset_index()
                    df_e.rename(columns={df_e.columns[0]: 'filename'}, inplace=True)
                    df_e['filename'] = df_e['filename'].apply(clean_filename)
                    data_e[(ds, art)] = df_e
            except Exception: 
                continue

        # Aggregate GLOBAL
        artifacts = sorted(list(set(k[1] for k in data_p.keys())))
        for art in artifacts:
            p_list = [data_p[(ds, art)] for ds in ['tcga', 'anorak'] if (ds, art) in data_p]
            e_list = [data_e[(ds, art)] for ds in ['tcga', 'anorak'] if (ds, art) in data_e]
            if p_list:
                data_p[('global', art)] = pd.concat(p_list, ignore_index=True)
                data_e[('global', art)] = pd.concat(e_list, ignore_index=True)

        for ds_key in ['tcga', 'anorak', 'global']:
            if (ds_key, 'base') not in data_p: 
                continue
            
            df_base_full = pd.merge(data_p[(ds_key, 'base')], data_e[(ds_key, 'base')], on='filename')
            f_cols = [c for c in df_base_full.columns if str(c).startswith('feature_')]

            for art in artifacts:
                if art == 'base' or (ds_key, art) not in data_p: 
                    continue
                
                df_art_full = pd.merge(data_p[(ds_key, art)], data_e[(ds_key, art)], on='filename')
                merged = pd.merge(df_base_full, df_art_full, on='filename', suffixes=('_b', '_a'))
                if merged.empty: 
                    continue

                pack = {
                    'y_true': merged['expected_label_b'].values,
                    'y_pred_b': merged['predicted_label_b'].values,
                    'y_pred_a': merged['predicted_label_a'].values,
                    'conf_b': merged[[c + '_b' for c in ALL_CLASSES]].max(axis=1).values,
                    'conf_a': merged[[c + '_a' for c in ALL_CLASSES]].max(axis=1).values,
                    'vec_b': merged[[c + '_b' for c in f_cols]].values,
                    'vec_a': merged[[c + '_a' for c in f_cols]].values
                }

                s = run_consistent_bootstrap(pack)
                
                full_results.append({
                    "Model": model_name, "Dataset": ds_key.upper(), "Artifact": art,
                    "Base_F1": round(s['f1_b'][0], 4), "Base_F1_Std": round(s['f1_b'][1], 4),
                    "Art_F1": round(s['f1_a'][0], 4), "Art_F1_Std": round(s['f1_a'][1], 4),
                    "Prec_Art": round(s['p_a'][0], 4), "Rec_Art": round(s['r_a'][0], 4),
                    "RPD": round(s['rpd'][0], 2), "Robustness_R": round(s['rob_r'][0], 2),
                    "Robustness_R_Std": round(s['rob_r'][1], 2),
                    "ECE_Delta": round(s['ece_d'][0], 4), "ECE_Delta_CI": f"[{round(s['ece_d'][2], 4)}, {round(s['ece_d'][3], 4)}]",
                    "PFR_Mean": round(s['pfr'][0], 2), "PFR_Std": round(s['pfr'][1], 2),
                    "NFR_Mean": round(s['nfr'][0], 2), "NFR_Std": round(s['nfr'][1], 2),
                    "CosSim_Mean": round(s['cos'][0], 4), "CosSim_Std": round(s['cos'][1], 4),
                    "EucDist_Mean": round(s['euc'][0], 4), "Softmax_Delta": round(s['sof_d'][0], 4)
                })

    final_df = pd.DataFrame(full_results)
    
    output_dir = os.path.dirname(output_path)
    if output_dir: 
        os.makedirs(output_dir, exist_ok=True)
    final_df.to_csv(output_path, index=False)
    
    print(f"Processing complete! Results saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process model predictions and generate bootstrap metrics.")
    parser.add_argument("--base-path", type=str, required=True, help="Path to the directory containing model results.")
    parser.add_argument("--output", type=str, default="LUAD_C.csv", help="Path to save the output CSV.")
    
    args = parser.parse_args()
    main(args.base_path, args.output)