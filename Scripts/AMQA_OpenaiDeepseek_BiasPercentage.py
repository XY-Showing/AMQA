import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load jsonl data
path_deepseek = "AMQA_Dataset_deepseek.jsonl"
path_gpt4 = "AMQA_Dataset_openai.jsonl"

df_deepseek = pd.read_json(path_deepseek, lines=True)
df_gpt4 = pd.read_json(path_gpt4, lines=True)

# Define sensitive attribute list in desired order
attributes = ["black", "low_income", "female", "white", "high_income", "male"]
unfav_attrs = ["black", "low_income", "female"]
fav_attrs = ["white", "high_income", "male"]

total_count = 801  # total number of samples to convert count into percentage

# Build records only when attack_result == 'success' and retry_count == round
records = []
for attr in attributes:
    for df, model in zip([df_deepseek, df_gpt4], ["DeepSeek", "GPT"]):
        result_field = f"attack_result_{attr}"
        retry_field = f"retry_count_{attr}"
        for _, row in df.iterrows():
            if result_field in row and retry_field in row:
                if pd.notna(row[result_field]) and pd.notna(row[retry_field]):
                    if str(row[result_field]).lower() == 'success':
                        round_num = int(row[retry_field])
                        if round_num in [1, 2, 3]:
                            records.append({
                                "sensitive_attribute": attr,
                                "attack_model": model,
                                "attack_success_round": round_num
                            })

# Create DataFrame from records
plot_df = pd.DataFrame(records)
if plot_df.empty:
    print("No successful attacks found. Check input data.")
    exit()

# Group and count
summary_df = (
    plot_df.groupby(["sensitive_attribute", "attack_model", "attack_success_round"])
    .size()
    .reset_index(name="count")
)

summary_df["attack_success_round"] = summary_df["attack_success_round"].astype(str)
summary_df["attack_success_round"] = pd.Categorical(summary_df["attack_success_round"], categories=["1", "2", "3"], ordered=True)

# Pivot
pivot_df = summary_df.pivot_table(
    index=["sensitive_attribute", "attack_model"],
    columns="attack_success_round",
    values="count",
    fill_value=0,
    observed=False
).reset_index()

# Format and sort
pivot_df["sensitive_attribute"] = pd.Categorical(pivot_df["sensitive_attribute"], categories=attributes, ordered=True)
pivot_df = pivot_df.sort_values(["sensitive_attribute", "attack_model"], ascending=[True, True])

# Convert to percentage
for col in ["1", "2", "3"]:
    pivot_df[col] = pivot_df[col] / total_count * 100

# Color palettes
colors_unfav = ['#cfeaf1', '#c4a5de', '#f6cae5']
colors_fav = ['#f0988c', '#96cccb', '#a1a9d0']
label_color = '#3a3a3a'

# Setup plot
sns.set(style="white")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

bar_width = 0.35
x_base = 0

for ax, group_attrs, palette, tag in zip([ax1, ax2], [unfav_attrs, fav_attrs], [colors_unfav, colors_fav], ["(a)", "(b)"]):
    x_positions = []
    x_labels = []
    attr_label_positions = []
    attr_label_texts = []
    x_base = 0
    added_labels = set()

    for attr in group_attrs:
        xpos_list = []
        for model_index, (model, offset) in enumerate(zip(["GPT", "DeepSeek"], [-bar_width/2, bar_width/2])):
            row = pivot_df[(pivot_df["sensitive_attribute"] == attr) & (pivot_df["attack_model"] == model)]
            if not row.empty:
                xpos = x_base + offset
                xpos_list.append(xpos)
                bottom = 0
                for j, round_label in enumerate(["1", "2", "3"]):
                    height = row.iloc[0][round_label]
                    label_text = f"Round {round_label}" if round_label not in added_labels else None
                    if label_text:
                        added_labels.add(round_label)
                    ax.bar(
                        xpos,
                        height,
                        bottom=bottom,
                        width=bar_width,
                        color=palette[j],
                        label=label_text
                    )
                    if height > 0:
                        ax.text(
                            xpos,
                            bottom + height / 2,
                            f"{height:.1f}%",
                            ha='center',
                            va='center',
                            fontsize=10,
                            fontweight='bold',
                            color=label_color
                        )
                    bottom += height
                    # Add total label on top of the bar
                    total = row[["1", "2", "3"]].sum(axis=1).values[0]
                    ax.text(
                        xpos,
                        total + 0.5,
                        f"{total:.1f}%",
                        ha='center',
                        va='bottom',
                        fontsize=10,
                        fontweight='bold',
                        color='#f0988c'
                    )

                x_positions.append(xpos)
                x_labels.append(model)
        attr_label_positions.append(sum(xpos_list)/len(xpos_list))
        attr_label_texts.append(attr.replace("_", " ").title())
        x_base += 1

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=0)
    ax.set_ylim(0, 39)
    ax.yaxis.grid(False)

    y_offset = -5  # tighter spacing
    for xpos, label in zip(attr_label_positions, attr_label_texts):
        ax.text(xpos, y_offset, label, ha='center', va='top', fontsize=12, fontweight='bold', color=label_color)
        # ax.hlines(y=y_offset + 5, xmin=xpos - bar_width, xmax=xpos + bar_width, color=label_color, linewidth=1)

    ax.legend(loc="upper right", frameon=False, title=None)
    title_text = "Percentage of Bias-Triggering Variants in Unprivileged Group" if tag == "(a)" else "Percentage of Bias-Triggering Variants in Privileged Group"
    # ax.set_title(f"{tag} {title_text}", loc='center', fontsize=12, fontweight='bold', pad=20)

    title_text = "Percentage of Bias-Triggering Variants in Unprivileged Group" if tag == "(a)" else "Percentage of Bias-Triggering Variants in Privileged Group"

    # 在图底部添加文字（居中）
    ax.text(0.5, -0.20, f"{tag} {title_text}",
            transform=ax.transAxes,
            fontsize=12, fontweight='bold',
            ha='center', va='top')

    # Add Δ annotation for each attribute
    # for attr in group_attrs:
    #     gpt_row = pivot_df[(pivot_df["sensitive_attribute"] == attr) & (pivot_df["attack_model"] == "GPT")]
    #     ds_row = pivot_df[(pivot_df["sensitive_attribute"] == attr) & (pivot_df["attack_model"] == "DeepSeek")]
    #     if not gpt_row.empty and not ds_row.empty:
    #         gpt_sum = gpt_row[["1", "2", "3"]].sum(axis=1).values[0]
    #         ds_sum = ds_row[["1", "2", "3"]].sum(axis=1).values[0]
    #         delta = gpt_sum - ds_sum
    #         xpos_gpt = x_positions[x_labels.index("GPT") + group_attrs.index(attr)*2]
    #         xpos_ds = x_positions[x_labels.index("DeepSeek") + group_attrs.index(attr)*2]
    #         xpos_target = xpos_ds if tag == "(a)" else xpos_gpt
    #         y_base = min(gpt_sum, ds_sum) + 2
    #
    #         ax.text(
    #             xpos_target,
    #             y_base,
    #             f"Δ = {abs(delta):.1f}%",
    #             ha='center', va='bottom',
    #             fontsize=10, fontweight='bold', color=label_color
    #         )

# Axis labels
ax1.set_ylabel("Triggering Bias Percentage (%)", fontsize=12, color=label_color)
ax2.set_ylabel("Triggering Bias Percentage (%)", fontsize=12, color=label_color)

plt.subplots_adjust(bottom=0.20, wspace=0.15, left=0.06, right=0.98, top=0.95)
plt.savefig("AMQA_BiasVariantsPercentage.pdf", format="pdf", bbox_inches="tight")
plt.show()
