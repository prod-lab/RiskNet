import matplotlib.pyplot as plt
import time
from risknet.run import pipeline
import os

start = time.time()

#pipeline.py returns auc, pr, recall, time
'''
Generate:
- Credit score model (cs): The only feature is credit score. Pandas loading uses first 10M records.
- Original Gangster model (OG): the legacy code from Rod's project. Pandas loading uses first 10M records.
- Parquet boost (p_boost): OG + using parquet data loading. Parquet uses the entire data file
- All optimizations added (all_in): OG + parquet + feature engineering. Parquet uses the entire data file
'''
cs = pipeline.pipeline(fe_enabled=False, baseline=True, p_true=False) #No FE, only feature is credit score, no parquet
og = pipeline.pipeline(fe_enabled=False, baseline=False, p_true=False) #Legacy code from Rod, no parquet
p_boost = pipeline.pipeline(fe_enabled=False, baseline=False, p_true=True) #OG + parquet
all_in = pipeline.pipeline(fe_enabled=True, baseline=False) #Using FE features + original Freddie Mac features

auc = [cs[0][2], og[0][2], all_in[0][2]]
pr = [cs[1][2], og[1][2], all_in[1][2]]
times = [cs[3], og[3], all_in[3]]

end = time.time()
elapsed = end - start

print("Time to run all 3 models and plot: " + str(round((elapsed / 60), 2)) + "minutes")
#About 25 minutes to run all 3 models

print(order)
print(auc)

fig, ax = plt.subplots()
rects = ax.bar(order, auc)
ax.bar_label(rects, padding=3)
ax.set_ylabel("AUC")
ax.set_title("AUC for first, second, and third iterations of models")
plt.savefig('graphs/aucs.png')
plt.show()

fig, ax = plt.subplots()
rects = ax.bar(order, pr)
ax.bar_label(rects, padding=3)
ax.set_ylabel("Precision")
ax.set_title("Precision for first, second, and third iterations of models")
plt.savefig('graphs/prs.png')
plt.show()

fig, ax = plt.subplots()
rects = ax.bar(order, times)
ax.set_ylabel("Time")
ax.set_title("Time to train for first, second, and third iterations of models")
plt.savefig('graphs/time.png')
plt.show()