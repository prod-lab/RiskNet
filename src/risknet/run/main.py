import matplotlib.pyplot as plt
import time
from risknet.run import pipeline
import seaborn as sns
import pandas as pd

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
- Credit score model (cs): The only feature is credit score. Loaded using pandas
- Original Gangster model (OG): the legacy code from Rod's project. Loaded using parquet
- All optimizations added (all_in): OG + parquet + feature engineering. Loaded using parquet
'''
#Currently loading all with parquet
#cs = pipeline.pipeline(fe_enabled=False, baseline=True, p_true=False)
#og = pipeline.pipeline(fe_enabled=False, baseline=False, p_true=False)
og_parquet = pipeline.pipeline(fe_enabled=False, baseline=False, p_true=True)
#all_in = pipeline.pipeline(fe_enabled=True, baseline=False, p_true=True)
'''
models_used = [cs, og, og_parquet, all_in]
order = ['credit_score', 'original', 'original_with_parquet', 'feature_eng']

aucs = auc(models_used)
prs = pr(models_used)
ts = times(models_used)

end = time.time()
elapsed = end - start

print("Time to run all " + str(len(order)) + " models and plot: " + str(round((elapsed / 60), 2)) + " minutes")
#About 25 minutes to run 3 models

models = pd.DataFrame(
    {'model': order,
     'auc': aucs,
     'prs': prs,
     'times': ts
    })

print(aucs)
print(models)

auc_bar = sns.barplot(data=models, x='model', y='auc')
auc_bar.bar_label(auc_bar.containers[0], fontsize=10)
plt.savefig('graphs/aucs.png')
plt.show()

pr_bar = sns.barplot(data=models, x='model', y='prs')
pr_bar.bar_label(pr_bar.containers[0], fontsize=10)
plt.savefig('graphs/prs.png')
plt.show()

fig, ax = plt.subplots()
rects = ax.bar(order, ts)
ax.bar_label(rects)
ax.set_ylabel("Time")
ax.set_title("Time to train for first, second, and third iterations of models")
plt.savefig('graphs/time.png')
plt.show()
'''