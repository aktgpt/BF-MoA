import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("stats/grit_stats.csv")

sns.scatterplot(x="grit_mean",y = "count_nuclei_mean",  hue="moa", data=df)
plt.show()
sns.boxplot(x="moa", y="grit_mean", data=df)
plt.xticks(rotation=70)
# plt.show()
# plt.figure()
# sns.violinplot(x="moa", y="grit_std", data=df)
# plt.xticks(rotation=70)

plt.figure()
sns.boxplot(x="moa", y="count_nuclei_mean", data=df)
plt.xticks(rotation=70)

# plt.figure()
# sns.violinplot(x="moa", y="count_nuclei_std", data=df)
# plt.xticks(rotation=70)
plt.show()

x = 1
