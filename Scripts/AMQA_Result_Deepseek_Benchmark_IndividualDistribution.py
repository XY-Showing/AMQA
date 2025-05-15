import json
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'Arial'})
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

# === Sensitive attribute pairs (good, bad, group label, good_label, bad_label) ===
sensitive_pairs = [
    ("test_model_answer_white", "test_model_answer_black", "Race", "White", "Black"),
    ("test_model_answer_high_income", "test_model_answer_low_income", "Socioeconomic Status", "High Income", "Low Income"),
    ("test_model_answer_male", "test_model_answer_female", "Gender", "Male", "Female")
]

# === Load JSONL to DataFrame ===
def load_model_df(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return pd.DataFrame(json.loads(line) for line in f)

model_dfs = {model: load_model_df(path) for model, path in model_files.items()}

# === Compute outcome distribution ===
records = []
for model, df in model_dfs.items():
    for good_col, bad_col, group, good_label, bad_label in sensitive_pairs:
        for _, row in df.iterrows():
            gold = row["answer_idx"]
            good_pred, bad_pred = row[good_col], row[bad_col]
            if good_pred == gold and bad_pred == gold:
                outcome = "Both Correct"
            elif good_pred == gold:
                outcome = f"Only Correct in {good_label} Variant"
            elif bad_pred == gold:
                outcome = f"Only Correct in {bad_label} Variant"
            else:
                outcome = "Both Incorrect"
            records.append({
                "model": model_name_mapping[model],
                "group": group,
                "outcome": outcome
            })

outcome_df = pd.DataFrame(records)
summary = outcome_df.value_counts().reset_index(name="count")
total = summary.groupby(["model", "group"])["count"].transform("sum")
summary["percentage"] = summary["count"] / total * 100

# === Plot ===
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
palette_pool = [
    ['#a1a9d0', '#b883d3', '#cfeaf1', '#f6cae5'],
    ['#f0988c', '#9e9e9e', '#c4a5de', '#96cccb'],
    ['#cfeaf1', '#a1a9d0', '#f0988c', '#b883d3']
]


for i, group in enumerate(summary["group"].unique()):
    ax = axes[i]
    subset = summary[summary["group"] == group]
    palette = ['#96cccb', '#f6cae5', '#c4a5de', '#cfeaf1']
    # get good/bad label for this group
    for g_col, b_col, g_name, g_label, b_label in sensitive_pairs:
        if g_name == group:
            good_label = g_label
            bad_label = b_label
            break

    # pivot to stack bars manually
    model_order = ["GPT-4.1", "Claude-3.7-Sonnet", "Gemini-2.0-Flash", "Qwen-Max", "Deepseek-V3"]
    pivot = subset.pivot(index="model", columns="outcome", values="percentage").fillna(0).reindex(model_order)

    outcome_order = [
        "Both Correct",
        "Both Incorrect",
        f"Only Correct in {bad_label} Variant",
        f"Only Correct in {good_label} Variant"
    ]
    pivot = pivot.reindex(columns=outcome_order)

    # pivot = pivot[[col for col in pivot.columns if "Correct" in col or "Incorrect" in col]]  # preserve order

    bottom = [0] * len(pivot)
    for j, outcome in enumerate(pivot.columns):
        color = palette[j % len(palette)]
        # bars = ax.bar(pivot.index, pivot[outcome], bottom=bottom, label=outcome, color=color)
        bars = ax.bar(pivot.index, pivot[outcome], bottom=bottom, label=outcome, color=color, width=0.5)

        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_y() + height/2, f"{height:.1f}%", ha='center', va='center', fontsize=10)
        bottom = [b + h for b, h in zip(bottom, pivot[outcome])]

    ax.set_title(f" ({chr(97 + i)}) Model Answer Distribution of {group} Attribute", pad=40)
    # ax.set_xlabel("Model")
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.tick_params(axis='x', labelrotation=25)
    ax.set_ylabel("")
    ax.tick_params(axis='y', which='both', labelleft=True)
    ax.set_ylim(0, 100)
    # plt.subplots_adjust(bottom=0.3)

    # place legend inside each subplot
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=9)

plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.99])
plt.savefig("AMQA_Deepseek_Benchmark_AnswerDistribution.pdf", format="pdf", bbox_inches="tight")
plt.show()
