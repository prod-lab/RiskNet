import matplotlib.pyplot as plt
import time
from risknet.run import pipeline
import seaborn as sns

start = time.time()

'''Define how to get AUC, PR, and Times from pipeline.pipeline() output'''
def auc(models):
    values = []
    for model in models:
        values.append(model[0][2])
    return values

def pr(models):
    values = []
    for model in models:
        values.append(model[1][2])
    return values

def times(models):
    values = []
    for model in models:
        values.append(model[3])
    return values

#pipeline.py returns auc, pr, recall, time
'''
Iterate to generate:
- Credit score model (cs): The only feature is credit score
- Original Gangster model (OG): the legacy code from Rod's project
- All optimizations added (all_in): OG + parquet + feature engineering
'''
#Currently loading all with parquet
cs = pipeline.pipeline(fe_enabled=False, baseline=True, p_true=True)
og = pipeline.pipeline(fe_enabled=False, baseline=False, p_true=True)
all_in = pipeline.pipeline(fe_enabled=True, baseline=False, p_true=True)

models_used = [cs, og, all_in]
order = ['credit_score', 'original', 'feature_eng']

aucs = auc(models_used)
prs = pr(models_used)
ts = times(models_used)

end = time.time()
elapsed = end - start

print("Time to run all 3 models and plot: " + str(round((elapsed / 60), 2)) + " minutes")
#About 25 minutes to run 3 models

fig, ax = plt.subplots()
rects = ax.bar(order, aucs)
ax.bar_label(rects, padding=3)
ax.set_ylabel("AUC")
ax.set_title("AUC for first, second, and third iterations of models")
plt.savefig('graphs/aucs.png')
plt.show()

fig, ax = plt.subplots()
rects = ax.bar(order, prs)
ax.bar_label(rects, padding=3)
ax.set_ylabel("Precision")
ax.set_title("Precision for first, second, and third iterations of models")
plt.savefig('graphs/prs.png')
plt.show()

fig, ax = plt.subplots()
rects = ax.bar(order, ts)
ax.set_ylabel("Time")
ax.set_title("Time to train for first, second, and third iterations of models")
plt.savefig('graphs/time.png')
plt.show()