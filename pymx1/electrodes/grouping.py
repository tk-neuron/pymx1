import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN


def _add_xyn(df_map: pd.DataFrame):
    df_map = df_map.copy()
    df_map['xn'] = (df_map.x / 17.5).astype(int)
    df_map['yn'] = (df_map.y / 17.5).astype(int)
    return df_map


def detect_groups(df_map: pd.DataFrame, eps=3, min_samples=9):
    """
    recommended param values:
    for feature maximization: eps=3, min_samples=9
    for neuronal unit 3*3: eps=2, min_samples=9
    """
    df_map = _add_xyn(df_map)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(df_map[['xn', 'yn']].values) 
    df_map['group'] = clustering.labels_
    return df_map
