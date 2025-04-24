

import scipy.stats as stats
import pandas as pd
import numpy as np
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from datetime import datetime
import os

def run_statistical_tests(kl_differences): 
    # 1. One-Sample t-Test
    kl_differences = np.array(kl_differences)
    t_stat, p_value_t = stats.ttest_1samp(kl_differences, 0)
    print(f"One-Sample t-Test: t-statistic = {t_stat:.0f}, p-value = {p_value_t:.4f}")
    
    # 2. Wilcoxon Signed-Rank Test (non-parametric)
    w_stat, p_value_w = stats.wilcoxon(kl_differences)
    print(f"Wilcoxon Signed-Rank Test: W-statistic = {w_stat:.0f}, p-value = {p_value_w:.4f}")
    
    # 3. Bootstrap Confidence Interval for the Mean
    n_bootstraps = 10000  # Number of bootstrap samples
    bootstrap_means = np.random.choice(kl_differences, (n_bootstraps, len(kl_differences)), replace=True).mean(axis=1)
    ci_lower, ci_upper = np.percentile(bootstrap_means, [2.5, 97.5])

    print(f"Bootstrap 95% Confdence Interval: [{ci_lower:.0f}, {ci_upper:.0f}]")
    return p_value_t, p_value_w, [ci_lower, ci_upper]
    
def make_kde_plot(data, model1, model2, smoothing = 5): 
    
    # Make plot
    rename_for_readability = {
        "OPT6_7B": "OPT 6.7B",
        "OPT2_7B": "OPT 2.7B", 
        "Llama31_8B": "Llama 3.1 8B",
        "Llama32_3B": "Llama 3.2 3B",
        "OPT125M": "OPT_125M",
        "OPT350M": "OPT350M", 
        "Qwen2_5_0_5B": "Qwen 2.5 0.5B",
        "Qwen2_5_3B": "Qwen 2.5 3B",
        "Gemma2_2B": "Gemma2 2B",
        "GPT2": "GPT-2",
        "GPT2Large": "GPT-2 Large"
    }
    
    fig= plt.figure(figsize=(8, 5))
    
    p_value_t, p_value_w, [ci_lower, ci_upper] = run_statistical_tests(data)
    t_text = "True" if p_value_t < 0.01 else "False"
    t_color = "green" if p_value_t < 0.01 else "red"
    w_text = "True" if p_value_w < 0.01 else "False"
    w_color = "green" if p_value_w < 0.01 else "red"
    
    sns.kdeplot(data=data, fill=True, color='blue')
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2)

    # Labels and title
    plt.xlabel("KL Difference")
    plt.ylabel("Density")
    
    if model1 in rename_for_readability: 
        model1 = rename_for_readability[model1]
    if model2 in rename_for_readability: 
        model2 = rename_for_readability[model2]
        
    plt.title(f"{model1} vs {model2}")
    
    plt.text(x=0.75, y=0.95, s=f"Num_Samples={int(len(data)*smoothing/2)}", transform=plt.gca().transAxes, fontsize=10, color='grey')  
    plt.text(x=0.75, y=0.90, s=f"Smoothing={smoothing}", transform=plt.gca().transAxes, fontsize=10, color='grey')
    plt.text(x=0.75, y=0.85, s="Prompt=None", transform=plt.gca().transAxes, fontsize=10, color='grey')
    plt.text(x=0.75, y=0.80, s=f"----- Stats -----", transform=plt.gca().transAxes, fontsize=10, color='blue')
    plt.text(x=0.75, y=0.75, s="T-Test p<0.01=", transform=plt.gca().transAxes, fontsize=10, color='grey')
    plt.text(x=0.905, y=0.75, s=t_text, transform=plt.gca().transAxes, fontsize=10, color=t_color)
    plt.text(x=0.75, y=0.70, s=f"W-Stat p<0.01=", transform=plt.gca().transAxes, fontsize=10, color='grey')
    plt.text(x=0.915, y=0.70, s=w_text, transform=plt.gca().transAxes, fontsize=10, color=w_color)
    plt.text(x=0.75, y=0.65, s=f"CI=[{ci_lower:.0f}, {ci_upper:.0f}]", transform=plt.gca().transAxes, fontsize=10, color='grey')
    
    # Show plot
    plt.tight_layout()
    plt.xlim(-4000, 4000)
   

    now = datetime.now()
    formatted_date = now.strftime("%B_%d_%H")
    os.makedirs(f"/scratch/mr7401/projects/meta_comp/plots/kde_{formatted_date}_Smoothing={smoothing}", exist_ok =True)
    
    plt.savefig(f"/scratch/mr7401/projects/meta_comp/plots/kde_{formatted_date}_Smoothing={smoothing}/{model1}_vs_{model2}.pdf", format = "pdf")
    plt.show()
    return fig

def mean(lst): 
    return sum(lst) / len(lst)

def compute_kl_difference(m1_m1_ll, m1_m2_ll, m2_m1_ll, m2_m2_ll):
    kl_diff = -(mean(m1_m1_ll) + mean(m1_m2_ll)) + (mean(m2_m1_ll) + mean(m2_m2_ll))
    return kl_diff
    
def load_log_likelihoods(model_name): 

    # paths = [
    #     f"/scratch/mr7401/log_likelihoods/{model_name}/log_likelihood_NEW.jsonl",
    #     f"/scratch/mr7401/log_likelihoods/{model_name}/log_likelihood_2000.jsonl",
    #     f"/scratch/mr7401/log_likelihoods_rerun_old/{model_name}/log_likelihood_NEW.jsonl", 
    #     f"/scratch/mr7401/log_likelihoods_rerun_old/{model_name}/log_likelihood_2000.jsonl"
    #     ]
    paths = [
        f"/scratch/mr7401/log_likelihoods_Truncation_Fixed/{model_name}/log_likelihood_NEW.jsonl",
        f"/scratch/mr7401/log_likelihoods_Truncation_Fixed/{model_name}/log_likelihood_2000.jsonl",
        ]
    
    valid_paths = []
    for path in paths: 
        if os.path.exists(path): 
            valid_paths.append(path)

            print(f"Including {path}...")
    
    return load_dataset("json", data_files= valid_paths, split="train", streaming=False)

def generate_kl_diffs(m1 = "OPT2_7B", m2 = "OPT6_7B", smoothing = 5): 
    if m1 == m2: 
        return []

    # Load LL Data
    print(f"\n\n\n Starting Plotting for {m1} and {m2}")
    m1_lls = load_log_likelihoods(m1)
    m2_lls = load_log_likelihoods(m2)

    # Convert to Pandas for join
    df1 = m1_lls.to_pandas()
    df2 = m2_lls.to_pandas()

    # Drop duplicates if any present 
    df1 = df1.drop_duplicates(subset=["generation_id"])
    df2 = df2.drop_duplicates(subset=["generation_id"])
    
    # Select only comparison models 
    df1=df1[df1["gen_source_model"].isin([m1,m2])]
    df2=df2[df2["gen_source_model"].isin([m1,m2])]

    # Merge on `generation_id` 
    merged_df = df1.merge(df2, on="generation_id", how="inner")

    # Checks: confirm all samples matched, and all samples with matching IDs have matching sequences
    assert merged_df[f"{m1}_ll"].isna().sum() ==0 
    assert merged_df[f"{m2}_ll"].isna().sum() ==0 
    #assert (merged_df["generation_x"] == merged_df["generation_y"]).sum() == len(merged_df)
    merged_df = merged_df[merged_df["generation_x"] == merged_df["generation_y"]]

    kl_divs = []
    
    # Make subsets for each model's generations for simpler sampling 
    m1_generations=merged_df[merged_df["gen_source_model_x"].isin([m1])]
    m2_generations=merged_df[merged_df["gen_source_model_x"].isin([m2])]

    n_batches = int(np.floor((len(merged_df)/smoothing)))
    
    for i in range(n_batches): 
        m1_sample = m1_generations.sample(smoothing)
        m1_m1_ll, m1_m2_ll = m1_sample[f"{m1}_ll"].tolist(), m1_sample[f"{m2}_ll"].tolist()
        
        m2_sample = m2_generations.sample(smoothing)
        m2_m1_ll, m2_m2_ll = m2_sample[f"{m1}_ll"].tolist(), m2_sample[f"{m2}_ll"].tolist()
        kl = compute_kl_difference(m1_m1_ll, m1_m2_ll, m2_m1_ll, m2_m2_ll)
        kl_divs.append(kl)
    return kl_divs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plots")
    args, unknown_args = parser.parse_known_args()

    models = ["OPT125M", "OPT350M", "OPT2_7B", "OPT6_7B", "Qwen2_5_0_5B","Qwen2_5_3B","Gemma2_2B","GPT2","GPT2Large","Llama31_8B","Llama32_3B"]
    for smoothing in [1,5,10]:
        for m1 in models: 
            for m2 in models: 
                if m1 != m2:
                    try:
                        kl_diffs = generate_kl_diffs(m1 = m1, m2=m2, smoothing=smoothing) 
                        plots = make_kde_plot(data = kl_diffs, model1 = m1, model2 = m2, smoothing=smoothing)
                    except Exception as e: 
                        print(f"EXCEPTION: {e}", flush = True)
