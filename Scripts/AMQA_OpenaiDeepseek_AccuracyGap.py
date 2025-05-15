import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define paths to JSONL files for each model
file_paths = {
    "openai": "AMQA_summary_batch_openai.jsonl",
    "deepseek": "AMQA_summary_batch_deepseek.jsonl"
}

# Load JSONL files into a dictionary of DataFrames
model_dataframes = {}

# Map internal short keys to display names
model_name_mapping = {
    "openai": "Variants from GPT-Agent",
    "deepseek": "Variants from Deepseek-Agent"
}

for short_name, path in file_paths.items():
    with open(path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
        display_name = model_name_mapping[short_name]
        model_dataframes[display_name] = pd.DataFrame(lines)

# Define six fine-grained categories and three coarse groups
fine_categories = [
    ("white_question", "White"),
    ("black_question", "Black"),
    ("male_question", "Male"),
    ("female_question", "Female"),
    ("high_income_question", "High Income"),
    ("low_income_question", "Low Income")
]
coarse_groups = [
    ("race", ["white_question", "black_question"]),
    ("gender", ["male_question", "female_question"]),
    ("income", ["high_income_question", "low_income_question"])
]

# Accuracy subplot
accuracy_records = []
for model, df in model_dataframes.items():
    for question_type, label in fine_categories:
        acc = df[df['question_type'] == question_type]['accuracy'].values[0]
        accuracy_records.append({
            "Model": model,
            "Attribute": label,
            "Accuracy": acc
        })
accuracy_df = pd.DataFrame(accuracy_records)

# Accuracy gap subplot
gap_records = []
for group_name, qtypes in coarse_groups:
    for model, df in model_dataframes.items():
        try:
            acc_1 = df[df['question_type'] == qtypes[0]]['accuracy'].values[0]
            acc_2 = df[df['question_type'] == qtypes[1]]['accuracy'].values[0]
            gap = abs(acc_1 - acc_2)
            gap_records.append({
                "Model": model,
                "Attribute": group_name.capitalize(),
                "Accuracy Gap": gap
            })
        except IndexError:
            print(f"Missing data for {model} in group {group_name}")

gap_df = pd.DataFrame(gap_records)

# Plot
sns.set(style='white')
fig, axes = plt.subplots(1, 2, figsize=(18, 6), gridspec_kw={'width_ratios': [2, 1]})
palette_accuracy = ["#cfeaf1", "#c4a5de"]
palette_gap = ["#f6cae5", "#96cccb"]  # Only need 2 colors for 2 models

# Barplot: Accuracy per Attribute
sns.barplot(data=accuracy_df, x="Attribute", y="Accuracy", hue="Model", ax=axes[0], palette=palette_accuracy)
# axes[0].set_title("(a) Accuracy by Demographic Group", fontsize=16, fontweight='bold')
axes[0].tick_params(labelsize=14)
axes[0].set_xlabel("", fontsize=14)
axes[0].set_ylabel("Accuracy", fontsize=14, color="#444444")
axes[0].legend(title="", fontsize=12, title_fontsize=12, frameon=False)
axes[0].set_ylim(0.0, 1.15)
for container in axes[0].containers:
    axes[0].bar_label(container, fmt='%.2f', fontsize=12, padding=3, fontweight='bold', color="#f0988c")

axes[0].text(0.5, -0.15, "(a) Accuracy by Demographic Group",
             transform=axes[0].transAxes,
             fontsize=16, fontweight='bold',
             ha='center', va='top')
# Barplot: Accuracy Gap per Group
sns.barplot(data=gap_df, x="Attribute", y="Accuracy Gap", hue="Model", ax=axes[1], palette=palette_gap)
# axes[1].set_title("(b) Accuracy Gap by Counterfactual Pair", fontsize=16, fontweight='bold')
axes[1].tick_params(labelsize=14)
axes[1].set_xlabel("")
axes[1].set_ylabel("Accuracy Gap", fontsize=14, color="#444444")
axes[1].legend(title="", fontsize=12, title_fontsize=12, frameon=False)
axes[1].set_ylim(0.0, 0.55)
for container in axes[1].containers:
    axes[1].bar_label(container, fmt='%.2f', fontsize=12, padding=3, fontweight='bold', color="#f0988c")
axes[1].text(0.5, -0.15, "(b) Accuracy Gap by Counterfactual Pair",
             transform=axes[1].transAxes,
             fontsize=16, fontweight='bold',
             ha='center', va='top')

plt.tight_layout()
plt.savefig("AMQA_OpenaiDeepseek_AccuracyGap.pdf", format="pdf", bbox_inches="tight")
plt.show()
