## PYTHON LIBRARIES
# Importing libraries
import sys
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm, tqdm_notebook
#import missingno as msno
import matplotlib.gridspec
from scipy import stats
import matplotlib.patches as mpatches
# Define some general functions
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from adjustText import adjust_text
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import datetime
import statsmodels
import scikit_posthocs as sp
from scipy import stats

## WORKING DIRECTORY
path = "../analysis/plots/"
new_day = datetime.datetime.now().strftime("%Y%m%d")

palette = {"Continuous": "#808080", "Intermittent" : "#D4D4D4", "Control":"#FFFFFF"}

dfmales = pd.read_csv("../analysis/df_males.csv")
dfmales.set_index("Unnamed: 0", inplace = True)
dffemales = pd.read_csv("../analysis/df_females.csv")
dffemales.set_index("Unnamed: 0", inplace = True)
metadatamales = pd.read_csv("../analysis/metadata_males.csv")
metadatamales.set_index("ID", inplace = True)
metadatafemales = pd.read_csv("../analysis/metadata_females.csv")
metadatafemales.set_index("ID", inplace = True)
mer_males = pd.merge(dfmales, metadatamales.loc[:, ["Treatment"]], 
                     left_index = True, right_index = True)
mer_females = pd.merge(dffemales, metadatafemales.loc[:, ["Treatment"]], 
                     left_index = True, right_index = True)

stats_males = pd.read_csv("../analysis/stats_males_maarouf.csv")
stats_males.set_index("Unnamed: 0", inplace = True)
stats_females = pd.read_csv("../analysis/stats_females_maarouf.csv")
stats_females.set_index("Unnamed: 0", inplace = True)

# Plot quality control of protein abundance
sub = mer_males.sort_values(["Treatment"])
intensities = sub.drop(columns = ["Treatment"]).T
palint = []

for e in sub["Treatment"]:
    palint.append(palette[e])
    
## Average intensities per sample
fig, ax = plt.subplots(figsize = (15, 4))
g = sns.boxplot(data = intensities, ax = ax, palette = palint)
plt.xticks(rotation = 90)
plt.xlabel("Protein abundance")
plt.ylabel("Normalized protein expression")
ax.yaxis.label.set_size(14)
ax.xaxis.label.set_size(14)

matplotlib.rcParams['pdf.fonttype'] = 42
plt.tight_layout()
plt.savefig(f"{path}/RawQC_protein_abundance_males_{new_day}.pdf")

# Plot quality control of protein abundance females
sub = mer_females.sort_values(["Treatment"])
intensities = sub.drop(columns = ["Treatment"]).T
palint = []

for e in sub["Treatment"]:
    palint.append(palette[e])
    
## Average intensities per sample females
fig, ax = plt.subplots(figsize = (15, 4))
g = sns.boxplot(data = intensities, ax = ax, palette = palint)
plt.xticks(rotation = 90)
plt.xlabel("Protein abundance")
plt.ylabel("Normalized protein expression")
ax.yaxis.label.set_size(14)
ax.xaxis.label.set_size(14)

matplotlib.rcParams['pdf.fonttype'] = 42
plt.tight_layout()
plt.savefig(f"{path}/RawQC_protein_abundance_females_{new_day}.pdf")

# Filter those where tukey is significant for both
stats_females = stats_females.loc[(stats_females['Tukey_Continuous']< 0.05) & 
                                  (stats_females['Tukey_Intermittent']< 0.05)]
stats_males = stats_males.loc[(stats_males['Tukey_Continuous']< 0.05) & 
                                  (stats_males['Tukey_Intermittent']< 0.05)]

cmap_MB = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#555555","white","#1AA7EC"])

fig, ax = plt.subplots(figsize = (2.5,4))
sns.heatmap(stats_males.loc[:, ["FoldChange_Continuous", 
                                "FoldChange_Intermittent"]].sort_values(by = ["FoldChange_Continuous"]),
                center = 1, cmap = cmap_MB, cbar_kws={'shrink': 0.5})
matplotlib.rcParams['pdf.fonttype'] = 42
plt.tight_layout()
plt.savefig("{}/MB_sigheatmap_males_{}.pdf".format(path, new_day))

fig, ax = plt.subplots(figsize = (2.5,5))
sns.heatmap(stats_females.loc[:, ["FoldChange_Continuous", 
                                "FoldChange_Intermittent"]].sort_values(by = ["FoldChange_Continuous"]),
                center = 1, cmap = cmap_MB, cbar_kws={'shrink': 0.5})
matplotlib.rcParams['pdf.fonttype'] = 42
plt.tight_layout()
plt.savefig("{}/MB_sigheatmap_females_{}.pdf".format(path, new_day))

def make_pca2(in_df, a, my_pal, wd_dir, top=100):
    b = a + 1
    cols = in_df.columns
    pca = PCA(n_components = 4)
    sorted_mean = in_df.mean(axis = 1).sort_values()
    select = sorted_mean.tail(top)
    in_df = in_df.loc[select.index.values]
    pca.fit(in_df)
    temp_df = pd.DataFrame()
    temp_df['pc_{}'.format(a+1)] = pca.components_[a]
    temp_df['pc_{}'.format(b+1)] = pca.components_[b]
    temp_df.index = cols
    
    print(pca.explained_variance_ratio_)
    temp_df['color'] = my_pal
    
    fig, ax = plt.subplots(figsize = (4,4))
    temp_df.plot(kind = 'scatter', x = 'pc_{}'.format(a+1), y = 'pc_{}'.format(b+1), s = 30, 
                 c = temp_df["color"], ax = ax, edgecolors="black")
    
    ax.set_title("PCA", size = 20)
    ax.set_xlabel("PC{}_{:.3f}%".format(a+1, pca.explained_variance_ratio_[a]*100), size = 18)
    ax.set_ylabel("PC{}_{:.3f}%".format(b+1, pca.explained_variance_ratio_[b]*100), size = 18)
    
    ax.yaxis.label.set_size(18)
    ax.xaxis.label.set_size(18)
    ax.tick_params(axis = "x", labelsize = 12)
    ax.tick_params(axis = "y", labelsize = 12)
    plt.style.context('seaborn-darkgrid')
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    matplotlib.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()
    
    plt.savefig("{}/PCA_PCcomp_{}_{}.pdf".format(wd_dir, str(a + 1), new_day))

my_pal = []
for e in mer_males["Treatment"].to_list():
    my_pal.append(palette[e])
make_pca2(mer_males.drop(columns = ["Treatment"]).T.fillna(0), 0, my_pal, path)


my_pal = []
for e in mer_females["Treatment"].to_list():
    my_pal.append(palette[e])
make_pca2(mer_females.drop(columns = ["Treatment"]).T.fillna(0), 0, my_pal, path)
