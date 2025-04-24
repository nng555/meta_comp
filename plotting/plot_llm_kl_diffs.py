
##### Loading Metadata
from datasets import load_dataset
import seaborn as sns 
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import os 
import argparse

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
    
def make_kde_plot(path): 
    md1 = load_dataset("json", data_files=path, split="train", streaming=False)

    # Remove any metadata columns for efficiency
    cols_to_remove = [x for x in md1.column_names if x not in ["id", "gen1", "gen2", "metric"]]
    if len(cols_to_remove) > 0:
        md1 = md1.remove_columns(cols_to_remove)
    
    # Make plot
    model1, model2 = path.split("/")[4].split("_vs_")
    rename_for_readability = {
        "OPT6_7B": "OPT 6.7B",
        "OPT2_7B": "OPT 2.7B", 
        "Llama31_8B": "Llama 3.1 8B",
        "Llama32_3B": "Llama 3.2 3B",
        "OPT125M": "OPT_125M",
        "OPT350M": "OPT350M" 
    }
    
    fig= plt.figure(figsize=(8, 5))
    
    data = md1["metric"]
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
    
    plt.text(x=0.75, y=0.95, s="Num_Samples=10K", transform=plt.gca().transAxes, fontsize=10, color='grey')
    plt.text(x=0.75, y=0.90, s="Smoothing=5", transform=plt.gca().transAxes, fontsize=10, color='grey')
    plt.text(x=0.75, y=0.85, s="Prompt=None", transform=plt.gca().transAxes, fontsize=10, color='grey')
    
    plt.text(x=0.78, y=0.80, s=f"----- Stats -----", transform=plt.gca().transAxes, fontsize=10, color='blue')
    plt.text(x=0.78, y=0.75, s="T-Test p<0.01=", transform=plt.gca().transAxes, fontsize=10, color='grey')
    plt.text(x=0.935, y=0.75, s=t_text, transform=plt.gca().transAxes, fontsize=10, color=t_color)
    plt.text(x=0.78, y=0.70, s=f"W-Stat p<0.01=", transform=plt.gca().transAxes, fontsize=10, color='grey')
    plt.text(x=0.945, y=0.70, s=w_text, transform=plt.gca().transAxes, fontsize=10, color=w_color)
    plt.text(x=0.78, y=0.65, s=f"CI=[{ci_lower:.0f}, {ci_upper:.0f}]", transform=plt.gca().transAxes, fontsize=10, color='grey')
    
    # Show plot
    plt.tight_layout()
    plt.savefig(f"/scratch/mr7401/projects/meta_comp/plots/kl_diff_{model1}_vs_{model2}.pdf", format = "pdf")
    return fig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plots")
    args, unknown_args = parser.parse_known_args()
    
    for file in os.listdir("/scratch/mr7401/meta_datasets_cpu/"): 
        path = f"/scratch/mr7401/meta_datasets_cpu/{file}/meta_dataset.jsonl"
        print(path)
        fig = make_kde_plot(path)
    
    for file in os.listdir("/scratch/mr7401/meta_datasets/"): 
        path = f"/scratch/mr7401/meta_datasets/{file}/meta_dataset.jsonl"
        print(path)
        fig = make_kde_plot(path)

