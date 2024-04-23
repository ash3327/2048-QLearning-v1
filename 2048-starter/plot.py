from matplotlib import pyplot as plt
from cmdargs import args
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

""" 
do plotting by running command:
    python snowfight-starter/plot.py -f 'qtable_202212190101/data.csv,qtable_202212190102/data.csv' 
        -o 'data.png' 
"""

df = pd.DataFrame()
if args.mode == 'rgb_array':
    args.mode = 'human'
assert args.mode in ('human', 'ai')
if args.mode == 'human':
    has_epsilon = False
    random_overlay = True
else:
    has_epsilon = True
    random_overlay = False
try:
    for i, file in enumerate(args.file.split(',')):
        subdf = pd.read_csv(file.strip())  # in case of user input having spaces
        subdf = subdf.loc[:, 'episode':]
        subdf.loc[:, 'id'] = i
        df = pd.concat([df, subdf], ignore_index=True)
except FileNotFoundError as fnfe:
    print(fnfe)
    print(f"File {args.file} is not found.")
    exit()

fig, axs = plt.subplots(nrows=3 if has_epsilon else 2, ncols=2,
                            figsize=(8, 9 if has_epsilon else 6), num=1, clear=True)
fig.tight_layout()

df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
df = df.sort_values(by=['episode', 'score'])
df = df.reset_index()
df = df.loc[:, 'episode':]
rolling = df.groupby('id').rolling(10, on='episode').mean().loc[:, 'episode':]
rolling = rolling.sort_values(by=['episode', 'score']).loc[:, 'episode':]


def plot_key(key: str, ax):
    mmin = rolling.groupby('episode', group_keys=False).apply(lambda x: min(x[key]))
    mmax = rolling.groupby('episode', group_keys=False).apply(lambda x: max(x[key]))
    mmean = rolling.groupby('episode', group_keys=False).apply(lambda x: x[key].mean())

    ax.plot(mmean)
    ax.fill_between(mmin.index, mmin, mmax, alpha=.3)


plot_key('score', axs[0, 0])
# 'best' is used as 'random_score' when self.random_overlay is True.
try:
    if random_overlay:
        plot_key('best', axs[0, 0])
        axs[0, 0].legend(['score', None, 'random', None])
    else:
        axs[0, 0].plot(df['good'].dropna())
        axs[0, 0].plot(df['best'].dropna())
        axs[0, 0].legend(['score', 'good', 'best'])
except KeyError:
    pass
df['score'].hist(ax=axs[0, 1], range=[-5, 100], bins=105)
if random_overlay:
    df['best'].hist(ax=axs[0, 1], range=[-5, 100], bins=105, alpha=0.5)
axs[0, 1].set_xlabel("score")
axs[0, 1].set_ylabel("frequency")

# 'good' is used as 'random_step' when self.random_overlay is True.
plot_key('step', axs[1, 0])
if random_overlay:
    plot_key('good', axs[1, 0])
    axs[1, 0].legend(['step', None, 'random', None])
df['step'].hist(ax=axs[1, 1], range=[-5, 2000], bins=100)
if random_overlay:
    df['good'].hist(ax=axs[1, 1], range=[-5, 2000], bins=100, alpha=0.5)
axs[1, 1].set_xlabel("step")
axs[1, 1].set_ylabel("frequency")
if has_epsilon:
    axs[2, 0].set_ylim([0, 1])
    df['epsilon'].plot(ax=axs[2, 0])
    df['epsilon'].hist(ax=axs[2, 1], bins=100)
    axs[2, 1].set_xlabel("epsilon")
    axs[2, 1].set_ylabel("frequency")
del rolling

plt.savefig(args.output_file)
plt.show()

