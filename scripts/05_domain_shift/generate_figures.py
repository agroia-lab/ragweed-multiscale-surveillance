#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Instituto de Investigaciones Agropecuarias (INIA), Chile
"""
Generate publication-quality figures for ragweed domain shift analysis.

Reads pre-computed embeddings and distance matrices, produces 6 figures.

Expected inputs in --data-dir (cross-collection embedding analysis of the
nine image collections):
    mmd_matrix.csv                  square MMD distance matrix (short names)
    generalization_risk_scores.csv  per-collection risk scores
    embeddings_reduced.csv          per-image reduced embeddings with
                                    'source_collection' and 'origin' columns

Usage:
    python scripts/05_domain_shift/generate_figures.py \
        --data-dir data/international_databases/ragweed_embeddings \
        --out-dir outputs/figures
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import squareform
import warnings
import argparse
import os
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────
_ap = argparse.ArgumentParser(description="Figures for the ragweed domain-shift analysis")
_ap.add_argument("--data-dir", default="data/international_databases/ragweed_embeddings",
                 help="Folder with mmd_matrix.csv, generalization_risk_scores.csv, embeddings_reduced.csv")
_ap.add_argument("--out-dir", default="outputs/figures", help="Output folder for PNG figures")
_args = _ap.parse_args()
DATA_DIR = _args.data_dir
OUT_DIR = _args.out_dir
os.makedirs(OUT_DIR, exist_ok=True)

# ── Short name mapping ────────────────────────────────────────────────────
SHORT = {
    'CL_Seba_Labelled':           'CL_Seba',
    '01_NorthDakota_Individual':  'ND_Indiv',
    '03_Michigan_3Season':        'MI_3Season',
    'CL_Alberto_Labelled':        'CL_Alberto',
    '07_WeedCrop_PrecAg':         'WeedCrop',
    '01_NorthDakota_Aerial':      'ND_Aerial',
    '05_Purdue_4Weed':            'Purdue',
    'CL_StaRosa_Unlabelled':      'CL_StaRosa',
    '06_USDA_WeedCube':           'WeedCube',
}

CHILEAN = {'CL_Seba', 'CL_Alberto', 'CL_StaRosa'}

# Colors
BLUE_CL = '#2171b5'
ORANGE_INT = '#e6550d'

def origin_color(name):
    return BLUE_CL if name in CHILEAN else ORANGE_INT

# ── Load data ─────────────────────────────────────────────────────────────
print("Loading data...")
mmd_df = pd.read_csv(f"{DATA_DIR}/mmd_matrix.csv", index_col=0)
risk_df = pd.read_csv(f"{DATA_DIR}/generalization_risk_scores.csv")
embed_df = pd.read_csv(f"{DATA_DIR}/embeddings_reduced.csv")

# Map source_collection to short names in embed_df
embed_df['short_name'] = embed_df['source_collection'].map(SHORT)
embed_df['origin'] = embed_df['origin']  # already has Chilean / International

labels = list(mmd_df.index)  # already short names from the CSV
n = len(labels)

print(f"  MMD matrix: {n}x{n}")
print(f"  Embeddings: {embed_df.shape[0]} images")
print(f"  Collections: {labels}")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1: Hierarchical Clustering Dendrogram
# ══════════════════════════════════════════════════════════════════════════
print("\n[1/6] Generating MMD dendrogram...")

mmd_mat = mmd_df.values.copy()
# Ensure perfect symmetry
mmd_mat = (mmd_mat + mmd_mat.T) / 2
np.fill_diagonal(mmd_mat, 0)
condensed = squareform(mmd_mat)

Z = linkage(condensed, method='ward')

fig, ax = plt.subplots(figsize=(10, 6))

# Color function: map each leaf to origin color
leaf_colors = {labels[i]: origin_color(labels[i]) for i in range(n)}

def link_color_func(k):
    """Color internal branches gray; leaves get origin color."""
    return '#555555'

dend = dendrogram(
    Z,
    labels=labels,
    ax=ax,
    leaf_rotation=25,
    leaf_font_size=12,
    above_threshold_color='#555555',
    color_threshold=0,
)

# Color the leaf labels by origin
xlbls = ax.get_xticklabels()
for lbl in xlbls:
    name = lbl.get_text()
    lbl.set_color(origin_color(name))
    lbl.set_fontweight('bold')

ax.set_title("Hierarchical Clustering by MMD Distance", fontsize=15, fontweight='bold', pad=12)
ax.set_ylabel("Ward Linkage Distance", fontsize=12)
ax.tick_params(axis='y', labelsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend
legend_handles = [
    mpatches.Patch(color=BLUE_CL, label='Chilean'),
    mpatches.Patch(color=ORANGE_INT, label='International'),
]
ax.legend(handles=legend_handles, fontsize=11, loc='upper right', framealpha=0.9)

plt.tight_layout()
fig.savefig(f"{OUT_DIR}/fig_mmd_dendrogram.png", dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close(fig)
print("  Saved fig_mmd_dendrogram.png")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2: Generalization Risk Bar Plot
# ══════════════════════════════════════════════════════════════════════════
print("\n[2/6] Generating generalization risk barplot...")

risk_sorted = risk_df.sort_values('Avg_MMD', ascending=True).reset_index(drop=True)
colors = [origin_color(db) for db in risk_sorted['Database']]
mean_mmd = risk_sorted['Avg_MMD'].mean()

fig, ax = plt.subplots(figsize=(9, 5.5))
bars = ax.barh(
    risk_sorted['Database'],
    risk_sorted['Avg_MMD'],
    color=colors,
    edgecolor='white',
    linewidth=0.5,
    height=0.65,
)

# Annotate with image count
for i, (_, row) in enumerate(risk_sorted.iterrows()):
    ax.text(
        row['Avg_MMD'] + 0.008,
        i,
        f"n={int(row['N_images'])}",
        va='center', ha='left', fontsize=10, color='#333333',
    )

ax.axvline(mean_mmd, color='#333333', linestyle='--', linewidth=1.2, alpha=0.7)
ax.text(mean_mmd + 0.005, n - 0.5, f'mean = {mean_mmd:.3f}',
        fontsize=10, color='#333333', va='bottom')

ax.set_xlabel("Average MMD to All Other Databases", fontsize=12)
ax.set_title("Generalization Risk Score by Database\n(Average MMD to All Others)",
             fontsize=14, fontweight='bold', pad=10)
ax.tick_params(axis='both', labelsize=11)
ax.set_xlim(0, risk_sorted['Avg_MMD'].max() + 0.08)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

legend_handles = [
    mpatches.Patch(color=BLUE_CL, label='Chilean'),
    mpatches.Patch(color=ORANGE_INT, label='International'),
]
ax.legend(handles=legend_handles, fontsize=11, loc='lower right', framealpha=0.9)

plt.tight_layout()
fig.savefig(f"{OUT_DIR}/fig_generalization_risk_barplot.png", dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close(fig)
print("  Saved fig_generalization_risk_barplot.png")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Network Graph
# ══════════════════════════════════════════════════════════════════════════
print("\n[3/6] Generating similarity network...")

try:
    import networkx as nx
except ImportError:
    print("  networkx not available, installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'networkx'])
    import networkx as nx

# Image counts per database (from risk_df)
img_counts = dict(zip(risk_df['Database'], risk_df['N_images']))

G = nx.Graph()
for lab in labels:
    G.add_node(lab)

MMD_THRESHOLD = 0.25
edge_data = []
for i in range(n):
    for j in range(i+1, n):
        mmd_val = mmd_mat[i, j]
        if mmd_val < MMD_THRESHOLD and mmd_val > 0:
            G.add_edge(labels[i], labels[j], weight=mmd_val)
            edge_data.append((labels[i], labels[j], mmd_val))

fig, ax = plt.subplots(figsize=(10, 8))

# Layout: spring with weight = inverse MMD for positioning
pos = nx.spring_layout(G, k=2.5, iterations=80, seed=42,
                       weight=None)

# If some nodes are isolated, ensure they appear
for lab in labels:
    if lab not in pos:
        pos[lab] = np.random.RandomState(hash(lab) % 2**31).rand(2) * 2 - 1

# Node sizes proportional to image count (scaled)
node_sizes = [max(200, img_counts.get(lab, 50) * 0.8) for lab in G.nodes()]
node_colors = [origin_color(lab) for lab in G.nodes()]

# Draw edges with width inversely proportional to MMD
if edge_data:
    edges = G.edges(data=True)
    edge_widths = []
    edge_colors = []
    for u, v, d in edges:
        w = d['weight']
        # Thicker = more similar (lower MMD)
        width = max(0.5, (MMD_THRESHOLD - w) / MMD_THRESHOLD * 6)
        edge_widths.append(width)
        edge_colors.append('#888888')

    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths,
                           edge_color=edge_colors, alpha=0.5)

# Draw edge MMD labels
for u, v, d in G.edges(data=True):
    x = (pos[u][0] + pos[v][0]) / 2
    y = (pos[u][1] + pos[v][1]) / 2
    ax.text(x, y, f"{d['weight']:.2f}", fontsize=8, ha='center', va='center',
            color='#555555', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.8))

nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                       node_color=node_colors, edgecolors='white', linewidths=1.5)

# Labels with background
for lab, (x, y) in pos.items():
    count = img_counts.get(lab, '?')
    ax.text(x, y - 0.12, f"{lab}\n(n={count})", fontsize=10, ha='center', va='top',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#cccccc', alpha=0.85))

ax.set_title("Database Similarity Network (edges: MMD < 0.25)",
             fontsize=14, fontweight='bold', pad=12)
ax.axis('off')

legend_handles = [
    mpatches.Patch(color=BLUE_CL, label='Chilean'),
    mpatches.Patch(color=ORANGE_INT, label='International'),
    Line2D([0], [0], color='#888888', linewidth=3, alpha=0.5, label='Thick = more similar'),
]
ax.legend(handles=legend_handles, fontsize=11, loc='lower left', framealpha=0.9)

plt.tight_layout()
fig.savefig(f"{OUT_DIR}/fig_mmd_network.png", dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close(fig)
print("  Saved fig_mmd_network.png")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 4: UMAP with Centroids
# ══════════════════════════════════════════════════════════════════════════
print("\n[4/6] Generating UMAP with centroids...")

# Color palette for 9 collections
palette = {
    'CL_Seba':     '#2171b5',
    'CL_Alberto':  '#6baed6',
    'CL_StaRosa':  '#08519c',
    'ND_Indiv':    '#e6550d',
    'ND_Aerial':   '#fd8d3c',
    'MI_3Season':  '#31a354',
    'WeedCrop':    '#756bb1',
    'Purdue':      '#de2d26',
    'WeedCube':    '#636363',
}

fig, ax = plt.subplots(figsize=(11, 8))

# Scatter all points
for sname in palette:
    mask = embed_df['short_name'] == sname
    sub = embed_df[mask]
    if len(sub) == 0:
        continue
    ax.scatter(
        sub['umap_x'], sub['umap_y'],
        c=palette[sname], s=8, alpha=0.35, label=f"{sname} (n={len(sub)})",
        rasterized=True,
    )

# Compute centroids
centroids = embed_df.groupby('short_name')[['umap_x', 'umap_y']].mean()

# Plot centroids as large stars
for sname, row in centroids.iterrows():
    ax.scatter(
        row['umap_x'], row['umap_y'],
        marker='*', s=350, c=palette[sname],
        edgecolors='black', linewidths=1, zorder=10,
    )
    ax.annotate(
        sname, (row['umap_x'], row['umap_y']),
        textcoords="offset points", xytext=(8, 8),
        fontsize=9, fontweight='bold', color=palette[sname],
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#cccccc', alpha=0.85),
        zorder=11,
    )

# Connect each centroid to its nearest neighbor centroid
centroid_names = list(centroids.index)
centroid_coords = centroids.values
for i, name_i in enumerate(centroid_names):
    dists = np.sqrt(np.sum((centroid_coords - centroid_coords[i])**2, axis=1))
    dists[i] = np.inf
    j = np.argmin(dists)
    ax.plot(
        [centroid_coords[i, 0], centroid_coords[j, 0]],
        [centroid_coords[i, 1], centroid_coords[j, 1]],
        '--', color='#333333', linewidth=1.0, alpha=0.6, zorder=5,
    )

ax.set_xlabel("UMAP-1", fontsize=12)
ax.set_ylabel("UMAP-2", fontsize=12)
ax.set_title("UMAP Embedding Space with Database Centroids", fontsize=14, fontweight='bold', pad=12)
ax.tick_params(axis='both', labelsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend outside
ax.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1.01, 1.0),
          borderaxespad=0, framealpha=0.9, markerscale=2.5)

plt.tight_layout()
fig.savefig(f"{OUT_DIR}/fig_umap_with_centroids.png", dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close(fig)
print("  Saved fig_umap_with_centroids.png")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 5: Active Learning Concept Diagram
# ══════════════════════════════════════════════════════════════════════════
print("\n[5/6] Generating active learning concept diagram...")

np.random.seed(42)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# --- LEFT PANEL: Before Active Learning ---
# Training domain cloud
n_train = 200
train_x = np.random.normal(-2, 0.8, n_train)
train_y = np.random.normal(0, 1.0, n_train)

# New domain cloud (shifted)
n_new = 150
new_x = np.random.normal(3, 0.9, n_new)
new_y = np.random.normal(0.5, 1.0, n_new)

ax1.scatter(train_x, train_y, c=BLUE_CL, s=20, alpha=0.5, label='Training domain (Chilean)')
ax1.scatter(new_x, new_y, c=ORANGE_INT, s=20, alpha=0.5, label='New domain (International)')

# Draw a gap annotation
ax1.annotate('', xy=(1.0, 0), xytext=(-0.5, 0),
             arrowprops=dict(arrowstyle='<->', color='#c0392b', lw=2))
ax1.text(0.25, -0.5, 'Domain Gap\n(High MMD)', ha='center', fontsize=11,
         color='#c0392b', fontweight='bold')

ax1.set_title("Before Active Learning", fontsize=13, fontweight='bold')
ax1.set_xlabel("Embedding Dimension 1", fontsize=11)
ax1.set_ylabel("Embedding Dimension 2", fontsize=11)
ax1.legend(fontsize=9, loc='upper left', framealpha=0.9)
ax1.set_xlim(-5, 6)
ax1.set_ylim(-4, 4)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.tick_params(labelsize=10)

# --- RIGHT PANEL: After Active Learning ---
ax2.scatter(train_x, train_y, c=BLUE_CL, s=20, alpha=0.5, label='Training domain (Chilean)')
ax2.scatter(new_x, new_y, c=ORANGE_INT, s=20, alpha=0.5, label='New domain (International)')

# Bridge samples — targeted from the gap region
n_bridge = 25
bridge_x = np.random.normal(0.5, 0.6, n_bridge)
bridge_y = np.random.normal(0.2, 0.7, n_bridge)
ax2.scatter(bridge_x, bridge_y, c='#27ae60', s=100, marker='*',
            edgecolors='black', linewidths=0.5, zorder=10,
            label='Bridge samples (active learning)')

# Draw reduced gap annotation
ax2.annotate('', xy=(0.9, -1.5), xytext=(0.1, -1.5),
             arrowprops=dict(arrowstyle='<->', color='#27ae60', lw=2))
ax2.text(0.5, -2.1, 'Reduced Gap\n(Lower MMD)', ha='center', fontsize=11,
         color='#27ae60', fontweight='bold')

ax2.set_title("After Active Learning", fontsize=13, fontweight='bold')
ax2.set_xlabel("Embedding Dimension 1", fontsize=11)
ax2.set_ylabel("Embedding Dimension 2", fontsize=11)
ax2.legend(fontsize=9, loc='upper left', framealpha=0.9)
ax2.set_xlim(-5, 6)
ax2.set_ylim(-4, 4)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.tick_params(labelsize=10)

fig.suptitle("Active Learning Bridges Domain Gaps", fontsize=15, fontweight='bold', y=1.02)
fig.text(0.5, 0.98, "Targeted sampling from underrepresented regions reduces MMD",
         ha='center', fontsize=11, fontstyle='italic', color='#555555')

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f"{OUT_DIR}/fig_active_learning_concept.png", dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close(fig)
print("  Saved fig_active_learning_concept.png")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 6: Origin Separation (Violin/Box plots)
# ══════════════════════════════════════════════════════════════════════════
print("\n[6/6] Generating origin separation plot...")

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), sharey=False)

if HAS_SEABORN:
    # Violin plots with seaborn
    sns.violinplot(data=embed_df, x='origin', y='umap_x', ax=ax1,
                   palette={'Chilean': BLUE_CL, 'International': ORANGE_INT},
                   inner='quartile', linewidth=1.2, saturation=0.85, cut=0)
    sns.violinplot(data=embed_df, x='origin', y='umap_y', ax=ax2,
                   palette={'Chilean': BLUE_CL, 'International': ORANGE_INT},
                   inner='quartile', linewidth=1.2, saturation=0.85, cut=0)
else:
    # Fallback: box plots
    chilean = embed_df[embed_df['origin'] == 'Chilean']
    intl = embed_df[embed_df['origin'] == 'International']

    bp1 = ax1.boxplot([chilean['umap_x'], intl['umap_x']],
                      labels=['Chilean', 'International'],
                      patch_artist=True, widths=0.5)
    bp1['boxes'][0].set_facecolor(BLUE_CL)
    bp1['boxes'][1].set_facecolor(ORANGE_INT)
    for b in bp1['boxes']:
        b.set_alpha(0.7)

    bp2 = ax2.boxplot([chilean['umap_y'], intl['umap_y']],
                      labels=['Chilean', 'International'],
                      patch_artist=True, widths=0.5)
    bp2['boxes'][0].set_facecolor(BLUE_CL)
    bp2['boxes'][1].set_facecolor(ORANGE_INT)
    for b in bp2['boxes']:
        b.set_alpha(0.7)

ax1.set_title("UMAP-1 (x)", fontsize=13, fontweight='bold')
ax1.set_ylabel("UMAP-x coordinate", fontsize=12)
ax1.set_xlabel("Origin", fontsize=12)
ax1.tick_params(axis='both', labelsize=11)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

ax2.set_title("UMAP-2 (y)", fontsize=13, fontweight='bold')
ax2.set_ylabel("UMAP-y coordinate", fontsize=12)
ax2.set_xlabel("Origin", fontsize=12)
ax2.tick_params(axis='both', labelsize=11)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Add sample counts
cl_n = len(embed_df[embed_df['origin'] == 'Chilean'])
intl_n = len(embed_df[embed_df['origin'] == 'International'])
fig.text(0.5, -0.02,
         f"Chilean: n={cl_n}  |  International: n={intl_n}",
         ha='center', fontsize=11, color='#555555')

fig.suptitle("Distribution of Embedding Coordinates by Origin",
             fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()
fig.savefig(f"{OUT_DIR}/fig_origin_separation.png", dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close(fig)
print("  Saved fig_origin_separation.png")


# ══════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ALL FIGURES GENERATED SUCCESSFULLY")
print("=" * 60)

import os
files = sorted([f for f in os.listdir(OUT_DIR) if f.startswith('fig_') and f.endswith('.png')])
for f in files:
    fpath = os.path.join(OUT_DIR, f)
    size_kb = os.path.getsize(fpath) / 1024
    print(f"  {f:45s} {size_kb:7.1f} KB")

print(f"\nOutput directory: {OUT_DIR}")
print(f"Total figures: {len(files)}")
