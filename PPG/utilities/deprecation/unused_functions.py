def get_clipping_pivots(s):
    peak_idx = signal.find_peaks(s,distance=30)[0]
    # trim the first 2 peaks and the last 2 peaks
    peak_idx = peak_idx[2:len(peak_idx)-2]
    pivot_list = []
    for i,j in zip(peak_idx[:-1],peak_idx[1:]):
        pivot_list.append(np.argmin(s[i:j])+i)
    # if len(pivot_list) == 0:
    #     return np.array([0,len(s)-1])
    if 0 not in pivot_list:
        pivot_list =[0]+pivot_list
    if len(s)-1 not in pivot_list:
        pivot_list = pivot_list + [len(s)-1]
    return np.array(pivot_list)
