from pandas import MultiIndex
import numpy as np


def reorder_multiindex(df, new_order_names):
    df = df.copy()
    old_index = df.index
    levels, labels, names = old_index.levels, old_index.labels, old_index.names
    i_new_order = [names.index(name) for name in new_order_names]
    new_levels = np.array(levels)[i_new_order]
    new_labels = np.array(labels)[i_new_order]
    new_names = np.array(names)[i_new_order]
    assert (len(np.intersect1d(new_order_names, names)) ==
            len(new_order_names) == len(names))
    df.index = MultiIndex(levels=new_levels,
                              labels=new_labels,
                              names=new_names)
    return df
