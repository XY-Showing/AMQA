import json
import pandas as pd
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
    "claude": "AMQA_Benchmark_Answer_claude_no_cot.jsonl",
    "deepseek": "AMQA_Benchmark_Answer_deepseek_no_cot.jsonl",
    "gemini": "AMQA_Benchmark_Answer_gemini_no_cot.jsonl",
    "openai": "AMQA_Benchmark_Answer_openai_no_cot.jsonl",
    "qwen": "AMQA_Benchmark_Answer_qwen_no_cot.jsonl"
}

# === Sensitive attributes ===
cf_pairs = [
    ("test_model_answer_white", "test_model_answer_black"),
    ("test_model_answer_high_income", "test_model_answer_low_income"),
    ("test_model_answer_male", "test_model_answer_female")
]

adv_attributes = [
    "test_model_answer_white", "test_model_answer_black",
    "test_model_answer_high_income", "test_model_answer_low_income",
    "test_model_answer_male", "test_model_answer_female"
]

# === Categorize answer change ===
def categorize_change(row, adv_col):
    adv_ans = row[adv_col]
    desens_ans = row["test_model_answer_desensitized_question"]
    if adv_ans == desens_ans:
        return "unchanged"
    elif (adv_ans == row["answer_idx"]) and not row["desens_correct"]:
        return "X→✓"
    elif (adv_ans != row["answer_idx"]) and row["desens_correct"]:
        return "✓→X"
    else:
        return "X→X"

# === Load and process each model ===
def process_model(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    df = pd.DataFrame(records)
    df["desens_correct"] = df["test_model_answer_desensitized_question"] == df["answer_idx"]
    for attr in adv_attributes:
        label = attr.replace("test_model_answer_", "")
        df[f"change_vs_desens_{label}"] = df.apply(lambda row: categorize_change(row, attr), axis=1)
    return df

model_results = {model: process_model(path) for model, path in model_files.items()}

# === Prepare summary dataframe ===
shift_summary = []
for model, df in model_results.items():
    for attr in adv_attributes:
        label = attr.replace("test_model_answer_", "")
        col = f"change_vs_desens_{label}"
        stats = df[col].value_counts().to_dict()
        for change_type in ["X→✓", "✓→X", "X→X"]:
            shift_summary.append({
                "model": model_name_mapping[model],
                "attribute": label,
                "change_type": change_type,
                "count": stats.get(change_type, 0)
            })

shift_df = pd.DataFrame(shift_summary)

# === Plot results ===
unique_attrs = shift_df["attribute"].unique()
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
axes = axes.flatten()

palettes = [
    {"X→✓": "#f0988c", "✓→X": "#c4a5de", "X→X": "#96cccb"},
    {"X→✓": "#9e9e9e", "✓→X": "#cfeaf1", "X→X": "#f6cae5"},
    {"X→✓": "#a1a9d0", "✓→X": "#96cccb", "X→X": "#b883d3"},
    {"X→✓": "#f6cae5", "✓→X": "#9e9e9e", "X→X": "#c4a5de"},
    {"X→✓": "#c4a5de", "✓→X": "#f0988c", "X→X": "#a1a9d0"},
    {"X→✓": "#b883d3", "✓→X": "#a1a9d0", "X→X": "#9e9e9e"}
]

subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
for i, attribute in enumerate(unique_attrs):
    subset = shift_df[(shift_df["attribute"] == attribute) & (shift_df["change_type"] != "unchanged")]
    palette = palettes[i % len(palettes)]
    model_order = [model_name_mapping[m] for m in ["openai", "claude", "qwen", "gemini", "deepseek"]]

    ax = axes[i]
    barplot = sns.barplot(data=subset, x="model", y="count", hue="change_type", ax=ax, palette=palette, order=model_order)
    max_count = subset['count'].max()
    ax.set_ylim(0, max(160, max_count + 20))
    for container in barplot.containers:
        barplot.bar_label(container, fmt='%d', label_type='edge', fontsize=8, padding=1)
        ax.set_title(f"{subplot_labels[i]} Answer Change: {attribute}")
    ax.set_ylabel("Count")
    ax.set_xlabel("Model")
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.99), frameon=False, fontsize=9)
plt.tight_layout(rect=[0.05, 0, 1, 0.95])
plt.show()
