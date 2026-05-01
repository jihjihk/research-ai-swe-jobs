#!/usr/bin/env python3
"""
T07: Power analysis visualization.
Creates a summary figure showing MDEs across all key comparisons.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_FIGURES = "exploration/figures/T07"
OUT_TABLES = "exploration/tables/T07"
os.makedirs(OUT_FIGURES, exist_ok=True)

# Load feasibility table
feas = pd.read_csv(f"{OUT_TABLES}/feasibility_summary.csv")

# Create a summary figure
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Panel 1: Binary MDE
ax = axes[0]
feas_plot = feas[feas['MDE_binary_p30'] != 'inf'].copy()
feas_plot['MDE_binary_p30'] = feas_plot['MDE_binary_p30'].astype(float)
feas_plot = feas_plot.sort_values('MDE_binary_p30')

colors = []
for v in feas_plot['verdict']:
    if v == 'well-powered':
        colors.append('green')
    elif v == 'adequate':
        colors.append('limegreen')
    elif v == 'marginal':
        colors.append('orange')
    else:
        colors.append('red')

labels = [f"{row['analysis_type']}\n({row['comparison']})" for _, row in feas_plot.iterrows()]
y_pos = np.arange(len(feas_plot))

bars = ax.barh(y_pos, feas_plot['MDE_binary_p30'], color=colors, alpha=0.7, edgecolor='gray', linewidth=0.5)

# Add MDE values as text
for i, (idx, row) in enumerate(feas_plot.iterrows()):
    ax.text(row['MDE_binary_p30'] + 0.005, i, f"{row['MDE_binary_p30']:.3f}", va='center', fontsize=7)

# Threshold lines
ax.axvline(0.03, color='green', linestyle='--', alpha=0.4, linewidth=0.8, label='Well-powered (<0.03)')
ax.axvline(0.05, color='orange', linestyle='--', alpha=0.4, linewidth=0.8, label='Adequate (<0.05)')
ax.axvline(0.10, color='red', linestyle='--', alpha=0.4, linewidth=0.8, label='Marginal (<0.10)')

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=7)
ax.set_xlabel('MDE (absolute proportion, p_base=0.30)')
ax.set_title('Binary Outcome MDE\n(80% power, alpha=0.05)')
ax.legend(fontsize=7, loc='lower right')

# Panel 2: Continuous MDE (Cohen's d)
ax = axes[1]
feas_plot2 = feas[feas['MDE_continuous_d'] != 'inf'].copy()
feas_plot2['MDE_continuous_d'] = feas_plot2['MDE_continuous_d'].astype(float)
feas_plot2 = feas_plot2.sort_values('MDE_continuous_d')

colors2 = []
for v in feas_plot2['verdict']:
    if v == 'well-powered':
        colors2.append('green')
    elif v == 'adequate':
        colors2.append('limegreen')
    elif v == 'marginal':
        colors2.append('orange')
    else:
        colors2.append('red')

labels2 = [f"{row['analysis_type']}\n({row['comparison']})" for _, row in feas_plot2.iterrows()]
y_pos2 = np.arange(len(feas_plot2))

ax.barh(y_pos2, feas_plot2['MDE_continuous_d'], color=colors2, alpha=0.7, edgecolor='gray', linewidth=0.5)

for i, (idx, row) in enumerate(feas_plot2.iterrows()):
    ax.text(row['MDE_continuous_d'] + 0.01, i, f"d={row['MDE_continuous_d']:.3f}", va='center', fontsize=7)

# Cohen's d benchmarks
ax.axvline(0.20, color='green', linestyle='--', alpha=0.4, linewidth=0.8, label='Small effect (d=0.20)')
ax.axvline(0.50, color='orange', linestyle='--', alpha=0.4, linewidth=0.8, label='Medium effect (d=0.50)')
ax.axvline(0.80, color='red', linestyle='--', alpha=0.4, linewidth=0.8, label='Large effect (d=0.80)')

ax.set_yticks(y_pos2)
ax.set_yticklabels(labels2, fontsize=7)
ax.set_xlabel("MDE (Cohen's d)")
ax.set_title("Continuous Outcome MDE\n(80% power, alpha=0.05)")
ax.legend(fontsize=7, loc='lower right')

plt.suptitle('T07: Statistical Power & Feasibility Summary', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT_FIGURES}/power_summary.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved power summary figure")


# Also create a sample size summary figure
fig, ax = plt.subplots(figsize=(10, 5))

# Group sizes by seniority variable and source
data = {
    'seniority_final': {
        'arshkon': {'entry': 848, 'associate': 241, 'mid-senior': 2924, 'director': 22, 'unknown': 984},
        'asaniczka': {'entry': 129, 'associate': 1451, 'mid-senior': 21542, 'director': 91, 'unknown': 0},
        'scraped': {'entry': 4656, 'associate': 996, 'mid-senior': 26805, 'director': 351, 'unknown': 2254},
    },
    'seniority_llm (labeled)': {
        'arshkon': {'entry': 84, 'associate': 15, 'mid-senior': 404, 'director': 1, 'unknown': 2629},
        'asaniczka': {'entry': 39, 'associate': 69, 'mid-senior': 498, 'director': 7, 'unknown': 2164},
        'scraped': {'entry': 180, 'associate': 30, 'mid-senior': 409, 'director': 30, 'unknown': 3100},
    },
    'seniority_native': {
        'arshkon': {'entry': 769, 'associate': 275, 'mid-senior': 2344, 'director': 25, 'unknown': 0},
        'asaniczka': {'entry': 0, 'associate': 2014, 'mid-senior': 21199, 'director': 0, 'unknown': 0},
        'scraped': {'entry': 3972, 'associate': 1156, 'mid-senior': 22872, 'director': 448, 'unknown': 0},
    },
}

# Create a grouped bar chart showing entry-level counts by source and seniority variable
sen_vars = ['seniority_final', 'seniority_llm (labeled)', 'seniority_native']
sources = ['arshkon', 'asaniczka', 'scraped']
levels = ['entry', 'associate', 'mid-senior', 'director']

x = np.arange(len(levels))
width = 0.08
offsets = np.linspace(-0.3, 0.3, len(sen_vars) * len(sources))

source_colors = {'arshkon': 'steelblue', 'asaniczka': 'coral', 'scraped': 'seagreen'}
var_hatches = {'seniority_final': '', 'seniority_llm (labeled)': '///', 'seniority_native': '...'}

i = 0
legend_entries = []
for sv in sen_vars:
    for src in sources:
        vals = [data[sv][src].get(lvl, 0) for lvl in levels]
        bars = ax.bar(x + offsets[i], vals, width, color=source_colors[src],
                      hatch=var_hatches[sv], alpha=0.7, edgecolor='gray', linewidth=0.5)
        i += 1

# Create legend manually
from matplotlib.patches import Patch
legend_elements = []
for src in sources:
    legend_elements.append(Patch(facecolor=source_colors[src], alpha=0.7, label=src))
for sv in sen_vars:
    legend_elements.append(Patch(facecolor='white', edgecolor='gray', hatch=var_hatches[sv], label=sv))

ax.set_xticks(x)
ax.set_xticklabels(levels)
ax.set_ylabel('Count (SWE postings)')
ax.set_title('Sample Sizes by Seniority Level, Source, and Seniority Variable')
ax.legend(handles=legend_elements, fontsize=7, ncol=2, loc='upper right')
ax.set_yscale('log')
ax.set_ylim(0.5, 50000)

plt.tight_layout()
plt.savefig(f"{OUT_FIGURES}/sample_sizes.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved sample sizes figure")
