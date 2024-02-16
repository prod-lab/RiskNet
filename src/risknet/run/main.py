import matplotlib.pyplot as plt
import time
from risknet.run import pipeline
import os

start = time.time()

order = ['baseline', 'original', 'fe']

#pipeline.py returns auc, pr, recall, time
first = pipeline.pipeline(fe_enabled=False, baseline=True) #No FE, only feature is credit score
second = pipeline.pipeline(fe_enabled=False, baseline=False) #No FE, using all original Freddie Mac features
third = pipeline.pipeline(fe_enabled=True, baseline=False) #Using FE features + original Freddie Mac features

auc = [first[0][2], second[0][2], third[0][2]]
pr = [first[1][2], second[1][2], third[1][2]]
times = [first[3], second[3], third[3]]

end = time.time()
elapsed = end - start

print("Time to run all 3 models and plot: " + str(round((elapsed / 60), 2)) + "minutes")

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