import pandas as pd
import numpy as np

# spike-data analysis
def channel_stats(df_sp: pd.DataFrame, df_map: pd.DataFrame):
    """
    calculate channel statistics (mean firing rate and spike amplitude for each channel)
    """
    duration = df_sp.spiketime.max() - df_sp.spiketime.min()
    groups = df_sp[['channel', 'amplitude']].groupby('channel')
    
    df_fr = pd.DataFrame(groups.size() / duration, columns=['firing_rate'])  # firing rate for each channel
    df_amp = groups.mean()  # mean spike amplitude for each channel
    
    df_stat = pd.concat([df_map.set_index('channel'), df_fr, df_amp], axis=1, join='inner')
    return df_stat