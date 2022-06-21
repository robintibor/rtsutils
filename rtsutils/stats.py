import numpy as np

def running_mean(arr, window_len, axis=0):
    # adapted from http://stackoverflow.com/a/27681394/1469195
    # need to pad to get correct first value also
    arr_padded = np.insert(arr,0,values=0,axis=axis)
    cumsum = np.cumsum(arr_padded,axis=axis)
    later_sums = np.take(cumsum, range(window_len, arr_padded.shape[axis]), 
        axis=axis)
    earlier_sums = np.take(cumsum, range(0, arr_padded.shape[axis] - window_len), 
        axis=axis)

    moving_average = (later_sums - earlier_sums) / float(window_len)
    return moving_average


def median(a, axis=None, keepdims=False):
    """
    Just since I use old numpy version on one cluster which doesn't
    have keepdims
    """
    out = np.median(a, axis)
    if keepdims:
        for ax in axis:
            out = np.expand_dims(out, ax)
    return out


def median_absolute_deviation(arr, axis=None, keepdims=False):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variability of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation 
        http://stackoverflow.com/a/23535934/1469195
    """
    arr = np.array(arr)
    if axis is None:
        axis = range(arr.ndim)
    med = median(arr, axis=axis, keepdims=True)
    return median(np.abs(arr - med), axis=axis, keepdims=keepdims)
