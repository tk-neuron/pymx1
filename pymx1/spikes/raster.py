import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def rasterplot(ax: plt.Axes, df: pd.DataFrame, start: float, end: float, x: str='spiketime', y: str='channel'):
    df_ = df.query('@start < spiketime < @end')
    return ax.scatter(x=df_[x], y=df_[y], color='black', s=5)

def spikehist(ax: plt.Axes, df: pd.DataFrame, start: float, end: float, bin_width: float, x: str='spiketime', smooth: bool=True):
    df_ = df.query('@start < spiketime < @end')
    hist, edges = np.histogram(df_[x].values, range=(start, end), bins=int((end - start) / bin_width))
    if smooth: hist = gaussian_filter(hist, sigma=[2], truncate=4)
    return ax.plot(edges[1:], hist, linewidth=2.0, color='black')

def rastergram(ax1: plt.Axes, ax2: plt.Axes, df: pd.DataFrame, start: float, end: float, bin_width: float=0.001, 
               x: str='spiketime', y: str='channel', smooth: bool=True, set_default_layout: bool=True):
    """
    fig, axes = plt.subplots(2, figsize=(10, 4), gridspec_kw={'height_ratios': [1, 3]})
    ax1, ax2 = axes
    rastergram(ax1=ax1, ax2=ax2, start=start, end=end)
    plt.subplots_adjust(hspace=0.5)
    plt.show()
    """
    p1 = spikehist(ax=ax1, df=df, start=start, end=end, bin_width=bin_width, x=x, smooth=smooth)
    p2 = rasterplot(ax=ax2, df=df, start=start, end=end, x=x, y=y)

    if set_default_layout:
        # layout for spikehist
        ax1.set_xlim(start, end)
        ax1.set_xticks([])
        ax1.set_ylabel('spikes')
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_linewidth(2)
        ax1.tick_params(width=2.0, length=5.0, direction='in')

        # layout for rasterplot
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['left'].set_linewidth(2)
        ax2.spines['bottom'].set_linewidth(2)

        # xaxis 
        ax2.set_xlim(start, end)
        ax2.set_xlabel('time [s]')

        for i, tick in enumerate(ax2.xaxis.get_ticklabels()):
            if i % 2 != 0:
                tick.set_visible(False) 

        # yaxis
        ax2.set_ylim(0, 1024)
        ax2.set_ylabel('channel #')
        ax2.set_yticks([0, 250, 500, 750, 1000])
        for i, tick in enumerate(ax2.yaxis.get_ticklabels()):
            if i % 2 != 0:
                tick.set_visible(False) 

        ax2.set_facecolor('whitesmoke')
        ax2.tick_params(width=2.0, length=5.0, direction='in')

    return p1, p2
