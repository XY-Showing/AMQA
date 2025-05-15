import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define paths to JSONL files for each model
file_paths = {
    "claude": "AMQA_D2_Benchmark_Summary_claude_no_cot.jsonl",
    "gemini": "AMQA_D2_Benchmark_Summary_gemini_no_cot.jsonl",
    "openai": "AMQA_D2_Benchmark_Summary_openai_no_cot.jsonl",
    "qwen": "AMQA_D2_Benchmark_Summary_qwen_no_cot.jsonl",
    "deepseek": "AMQA_D2_Benchmark_Summary_deepseek_no_cot.jsonl",
}

# Load JSONL files into a dictionary of DataFrames
model_dataframes = {}

# Map internal short keys to display names
model_name_mapping = {
    "claude": "Claude-3.7-Sonnet",
    "gemini": "Gemini-2.0-Flash",
    "openai": "GPT-4.1",
    "qwen": "Qwen-Max",
    "deepseek": "Deepseek-V3"
}

for short_name, path in file_paths.items():
    with open(path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
        display_name = model_name_mapping[short_name]
        model_dataframes[display_name] = pd.DataFrame(lines)

# Define categories to compare
categories = [
    ("white_question", "black_question", "White vs Black"),
    ("male_question", "female_question", "Male vs Female"),
    ("high_income_question", "low_income_question", "High Income vs Low Income"),
    ("original_question_question", "desensitized_question_question", "Original vs Neutralized")
]

# Set up 2x2 subplot grid
sns.set(style='white')
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()

# Iterate over each comparison category
# Assumes all models have complete data for all question types
for idx, (cat1, cat2, title) in enumerate(categories):
    # subplot_label = chr(ord('a') + idx)  # 'a', 'b', 'c', 'd'
    comp_data = []
    for model, df in model_dataframes.items():
        try:
            acc1 = df[df['question_type'] == cat1]['accuracy'].values[0]
            acc2 = df[df['question_type'] == cat2]['accuracy'].values[0]
            comp_data.append({
                'model': model,
                'cat1_accuracy': acc1,
                'cat2_accuracy': acc2
            })
        except IndexError:
            print(f"Missing data for {model} in category pair: {cat1}, {cat2}")

    comp_df = pd.DataFrame(comp_data)
    comp_df = comp_df[['model', 'cat1_accuracy', 'cat2_accuracy']]
    comp_df['accuracy_gap'] = (comp_df['cat1_accuracy'] - comp_df['cat2_accuracy']).abs()
    model_order = ["GPT-4.1", "Claude-3.7-Sonnet", "Gemini-2.0-Flash", "Qwen-Max", "Deepseek-V3"]
    comp_df = comp_df.set_index('model').loc[model_order].reset_index()

    bar_width = 0.3
    index = range(len(comp_df))
    ax = axs[idx]
    # ax.text(-0.14, 1.05, f'({subplot_label})', transform=ax.transAxes, fontsize=16, fontweight='bold', va='top')

    cat1_colors = ['#a1a9d0', '#b883d3', '#cfeaf1', '#f6cae5']  # soft sci palette, cat1
    ax.bar(index, comp_df['cat1_accuracy'], bar_width,
           label=cat1.replace("_question", "").replace("_", " ").replace("desensitized", "Neutralized").title(),
           alpha=0.9, color=cat1_colors[idx])
    cat2_colors = ['#f0988c', '#9e9e9e', '#c4a5de', '#96cccb']  # soft sci palette, cat2  # last color estimated
    ax.bar([i + bar_width for i in index], comp_df['cat2_accuracy'], bar_width,
           label=cat2.replace("_question", "").replace("_", " ").replace("desensitized", "Neutralized").title(),
           alpha=0.9, color=cat2_colors[idx])

    # ax.set_xlabel('Model', color='#444444', fontweight='bold')
    ax.set_ylabel('Accuracy', color='#444444', fontweight='bold')
    # ax.set_title(f'({chr(97 + idx)}) Accuracy: {title}', color='#444444', fontweight='bold')
    ax.set_xticks([i + bar_width / 2 for i in index])
    ax.set_xticklabels(comp_df['model'], rotation=25, color='#444444')
    ax.tick_params(axis='both', labelsize=15)

    ax.set_ylim(0.4, 1)
    # ax.legend(fontsize=14, frameon=False)
    ax.legend(loc='upper left', bbox_to_anchor=(0.64, 0.93), fontsize=14, frameon=False)
    ax.text(0.5, -0.35, f'({chr(97 + idx)}) Accuracy: {title}',
                 transform=ax.transAxes,
                 fontsize=16, fontweight='bold',
                 ha='center', va='top')
    # ax.grid(axis='y')  # removed per request

    ax2 = ax.twinx()
    ax2.tick_params(axis='y', labelsize=15)
    # Add accuracy gap line and annotate values
    ax2.plot([i + bar_width / 2 for i in index], comp_df['accuracy_gap'], color='#5e6472', marker='o', linestyle='None',
             linewidth=2, label='Accuracy Gap')
    ax2.set_ylabel('Accuracy Gap', color='#444444', fontweight='bold')
    ax2.set_ylim(0, 0.5)
    # Annotate each point with gap value
    for i, gap in enumerate(comp_df['accuracy_gap']):
        ax2.text(i + bar_width / 2, gap + 0.01, f"{gap:.3f}", color='#5e6472', ha='center', va='bottom', fontsize=16,
                 fontweight='bold')
    ax2.legend(loc='upper right', fontsize=14, frameon=False)

plt.tight_layout()
plt.savefig("AMQA_Openai_Benchmark_Accuracy.pdf", format="pdf", bbox_inches="tight")
plt.show()
