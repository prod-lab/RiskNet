import matplotlib.pyplot as plt
import time
from risknet.run import pipeline

start = time.time()

auc = []
pr = []
recall = []
times = []

def get_val_metric(output, auc, pr, recall, times):
    a, p, r, t = output
    auc.append(a[2])
    pr.append(p[2])
    recall.append(r[2])
    times.append(t)

order = ['baseline', 'original', 'fe']

#pipeline.py returns auc, pr, recall, time
first = pipeline.pipeline(fe_enabled=False, baseline=True) #No FE, only feature is credit score
second = pipeline.pipeline(fe_enabled=False, baseline=False) #No FE, using all original Freddie Mac features
third = pipeline.pipeline(fe_enabled=True, baseline=False) #Using FE features + original Freddie Mac features

get_val_metric(first)
get_val_metric(second)
get_val_metric(third)

plt.bar(order, auc)
plt.show()

plt.bar(order, pr)
plt.show()

plt.bar(order, recall)
plt.show()

plt.bar(order, time)
plt.show()

end = time.time()
elapsed = end - start

print("Time to run all 3 models and plot: " + str(round((elapsed / 60), 2)) + "minutes")