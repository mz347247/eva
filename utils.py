import pandas as pd
import numpy as np
import math
import multiprocessing.pool
import matplotlib.pyplot as plt

class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass

class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class NestablePool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(NestablePool, self).__init__(*args, **kwargs)

def interval_filter(df, min_time=None, min_volume=None, min_amount=None, strict=True):
    """
    filter out consecutive opportunities if the arrival time of the opportunities are too close

    df should have columns time and cum_volume
    """
    records = df.to_records(index=False)
    remaining = [records[0]]

    volume_cond = True
    time_cond = True
    amount_cond = True

    for oppo in records:
        if min_time is not None:
            time_cond = ((oppo.time - remaining[-1].time) / 1e6 > min_time)

        if min_volume is not None:
            volume_cond = (oppo.cum_volume - remaining[-1].cum_volume > min_volume)

        if min_amount is not None:
            amount_cond = (oppo.cum_amount - remaining[-1].cum_amount > min_amount)

        cond = time_cond
        if strict:
            cond = cond and (volume_cond and amount_cond)
        else:
            cond = cond and (volume_cond or amount_cond)

        if cond:
            remaining.append(oppo)

    return pd.DataFrame(np.array(remaining))


def find_top_percent(df, col, target_number, target_ratio, target_number_col, target_return_col, ytrue_col,
                     total_number, filter_first, min_time=1, min_volume=1500, min_amount=15000, strict=True,
                     tolerance=0.05, termination=20):
    """
    find out the top x percent opportunities so that there are target number of opportunities remaining after filtering
    """
    assert (target_ratio is not None) or (target_number is not None) or (target_return_col is not None)

    df_valid = df[df[col].notna() & df[ytrue_col].notna() & (~df['nearLimit'])]
    if len(df_valid) < 10:
        return None

    if target_return_col is not None:
        target_return = df_valid[target_return_col].iloc[0]
        if np.isnan(target_return):
            return None

        target_return = max(2e-4, target_return)
        low = 10
        high = len(df_valid)

        for _ in range(termination):
            number = low + (high - low) // 2
            oppo_index = df_valid.index[df_valid[col] >= df_valid[col].quantile(1 - number / len(df_valid))]
            oppo = interval_filter(df_valid.loc[oppo_index], min_time, min_volume, min_amount, strict)
            vw_ret = weighted_average(oppo[ytrue_col], oppo['availNtl'])
            if abs(vw_ret - target_return) < tolerance * abs(target_return):
                break
            if high - low == 1:
                break
            elif vw_ret < target_return:
                high = number
            else:
                low = number

    else:
        if target_ratio is not None:
            if filter_first:
                df_pass_filter = interval_filter(df_valid, min_time, min_volume, min_amount, strict)
                target_number = len(df_pass_filter) * target_ratio
            elif target_number_col is not None:
                target_number = df_valid[target_number_col].iloc[0] * target_ratio
            else:
                target_number = int(total_number * target_ratio)

        target_number = math.ceil(target_number * (1 - (df.loc[df[col].isna() | df['nearLimit'], 'time']//1e6).nunique() / total_number))
        # exclude some special cases
        if (target_number <= 0) or (len(df_valid) < target_number):
            return None

        # pick the top x percentile first
        ratio = target_number / len(df_valid)
        filter_rate = 0

        for _ in range(termination):
            oppo_index = df_valid.index[df_valid[col] > df_valid[col].quantile(1 - ratio)]
            oppo = interval_filter(df_valid.loc[oppo_index], min_time, min_volume, min_amount, strict)

            if abs(len(oppo) - target_number) < tolerance * target_number:
                break
            
            # update the estimation of the filter rate
            filter_rate = 1/3 * filter_rate + 2/3 * len(oppo) / len(oppo_index)
            # update the ratio
            ratio = min(target_number / filter_rate / len(df_valid), 1)
        
    # if len(oppo) < 50:
    #     return None
        
    oppo['top_percent'] = len(oppo) / len(df_valid)
    return oppo

def weighted_average(values, weights):
    indices = ~np.isnan(values)
    return np.average(values[indices], weights=weights[indices])

def findTmValue(clockLs, tm, method='L'):
    maxIx = len(clockLs)
    orignIx = np.arange(maxIx)
    if method == 'F':
        ix = np.searchsorted(clockLs, clockLs + tm, side='left')
        ## if target future index is next tick, mask
        mask = (orignIx == ix)|(ix == maxIx)
    elif method == 'L':
        ## if target future index is last tick, mask
        ix = np.searchsorted(clockLs, clockLs - tm, side='right') - 1
        ix = ix - 1
        ix[ix<0] = 0
        mask = (orignIx == ix) | ((clockLs-tm).values < clockLs.values[0])
    ix[mask] = -1
    return ix

# from Andy
def render_mpl_table(data, col_width=2.6, row_height=0.8, font_size=15,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0, seperate_col = 6,
                     ax=None, **kwargs):
    import six
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, cellLoc='center', **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in  six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] in [0,seperate_col] or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors)])
    return ax

# from Changjian
def df_to_plt_table(data, width = 30, row_height = 0.8, font_size = 12, header_color = '#40466e',
                    row_colors = ['#f1f1f2', 'w'], edge_color = 'w', bbox = [0,0,1,1], header_columns = 0,
                    separate_col = 10, ax = None, **kwargs):
    # 此函数可将一个DataFrame转成matplotlib的表格，进而打印在PDF中
    # data: the DataFrame
    # width: the width of the table. Should be the same with the width of the pdf in the following part
    # separate_col: if this is 10, then for every 10 rows, there will be one row with different colour.
    import six
    if ax is None:
        size = (width, (data.shape[0]+1) * row_height)
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, cellLoc='center', rowLabels = data.index, rowLoc = 'center', **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in  six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] in [0,separate_col] or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors)])
    return ax 