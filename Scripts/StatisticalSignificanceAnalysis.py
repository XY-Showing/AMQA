import json
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
import matplotlib.pyplot as plt
import seaborn as sns


# === File mapping for each model ===
model_name_mapping = {
    "claude": "Claude-3.7-Sonnet",
    "gemini": "Gemini-2.0-Flash",
    "openai": "GPT-4.1",
    "qwen": "Qwen-Max",
    "deepseek": "Deepseek-V3"
}

model_files = {
    "claude": "AMQA_D2_Benchmark_Answer_claude_no_cot.jsonl",
    "deepseek": "AMQA_D2_Benchmark_Answer_deepseek_no_cot.jsonl",
    "gemini": "AMQA_D2_Benchmark_Answer_gemini_no_cot.jsonl",
    "openai": "AMQA_D2_Benchmark_Answer_openai_no_cot.jsonl",
    "qwen": "AMQA_D2_Benchmark_Answer_qwen_no_cot.jsonl"
}

sensitive_pairs = [
    ("test_model_answer_original_question", "test_model_answer_desensitized_question", "Baseline", "Original", "Neutralized"),
    ("test_model_answer_white", "test_model_answer_black", "Race", "White", "Black"),
    ("test_model_answer_high_income", "test_model_answer_low_income", "Socioeconomic Status", "High Income", "Low Income"),
    ("test_model_answer_male", "test_model_answer_female", "Gender", "Male", "Female")
]

# === Load JSONL file ===
def load_jsonl(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return pd.DataFrame(json.loads(line) for line in f)

# === Define McNemar test ===
def run_mcnemar(df, answer_col, variant_a, variant_b):
    b = c = 0
    for _, row in df.iterrows():
        correct = row[answer_col]
        pred_a = row.get(variant_a)
        pred_b = row.get(variant_b)

        a_correct = pred_a == correct
        b_correct = pred_b == correct

        if a_correct and not b_correct:
            b += 1
        elif not a_correct and b_correct:
            c += 1

    table = [[0, b], [c, 0]]
    result = mcnemar(table, exact=True)
    return b, c, result.pvalue

# === Format significance level ===
def significance_level(p):
    if p < 0.001:
        return "*** Very significant"
    elif p < 0.01:
        return "** Significant"
    elif p < 0.05:
        return "* Marginally significant"
    else:
        return "ns Not significant"

# === Main analysis ===
def analyze_all_models():
    results = []
    for model_key, file_path in model_files.items():
        df = load_jsonl(file_path)
        for col_a, col_b, group, label_a, label_b in sensitive_pairs:
            b, c, pval = run_mcnemar(df, "answer_idx", col_a, col_b)
            results.append({
                "Model": model_name_mapping[model_key],
                "Counterfactual Pair": f"{label_a} vs {label_b} ({group})",
                "b (Good correct, Bad wrong)": b,
                "c (Good wrong, Bad correct)": c,
                "p-value": pval,
                "Significance": significance_level(pval)
            })
    return pd.DataFrame(results)


# === Visualization ===
def plot_significance_bars(df):
    plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid")

    df_sorted = df.sort_values(by=["Counterfactual Pair", "Model"])

    ax = sns.barplot(
        data=df_sorted,
        x="Counterfactual Pair",
        y="p-value",
        hue="Model"
    )

    # Add significance stars on top
    for i, row in df_sorted.iterrows():
        sig = row["Significance"].split()[0]  # "***", "**", etc.
        if sig != "ns":
            ax.text(i, row["p-value"] + 0.02, sig, ha='center', va='bottom', fontsize=10, color='red')

    plt.xticks(rotation=45, ha="right")
    plt.yscale("log")
    plt.ylabel("p-value (log scale)")
    plt.title("Statistical Significance of Model Bias across Counterfactual Pairs")
    plt.tight_layout()
    plt.savefig("mcnemar_significance_plot.png")
    plt.show()


# === Example usage ===
if __name__ == "__main__":
    result_df = analyze_all_models()
    plot_significance_bars(result_df)

    # result_df.to_csv("mcnemar_all_models_results.csv", index=False)
    print(result_df)
