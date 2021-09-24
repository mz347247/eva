# **Table of contents**
1. [Usage](#usage)
2. [Configuration Parameters](#configuration-parameters)
3. [Evaluation Methods in Detail](#evaluation-methods-in-detail)
4. [Intermediate Evaluation Statistics](#intermediate-evaluation-statistics)
5. [Evaluation Report](#evaluation-report)

## **Usage**
1. Clone into this repo! Use the following ssh otherwise it's slow. 

    ``git@github.com:study-int/sta_evaluation_equities_cn.git``
2. Modify the .yaml configuration file
3. Run from the command line
    
    ``python eval_run.py <configuration file path>``

4. Check the evaluation report saved in the assigned `save_path` in the configuration file

### Requirements
* python >= 3.6
* pandas >= 0.25.0
* plotly >= 5.3.0
* update your dfs api to jdfs >= 1.1.0. You can set up the dfs following the instruction in this [link](https://github.com/study-int/dfs_sh_client).
* make sure you can use git via ssh keys


### Example
``python eval_run.py sta_input_ps_example.yaml``

## **Configuration Parameters**:
Two example configuration files is shown in `sta_input_ps_example.yaml` and ``sta_input_hpc_example.yaml`
 
### **machine** :  ***{personal-server, HPC}***
The machine to run the evaluation program

### **bench** : ***{IC, IF}***
The bench index to read the md and return data

### **universe** : ***{custom, IC, IF}***
The set of stocks to run the evaluation program on. If custom, will use all the stocks under the specified directory.

### **njobs** : ***int***
The number of processors on personal server or the number of jobs on HPC used to run the evaluation program.

### **start_date** : ***int***
Start date of the evaluation period

### **end_date** : ***int***
End date of the evaluatin period

### **eval_alpha** :
A list of data sources. Each source can contain multiple alphas or just one alpha.
    
> **name** : ***str*** <br> The name of this data source. Will be used to distinguish alphas from different data sources and also shown in the evaluation report.

> **data_type** : ***{l2, mbd}*** <br> Whether the alpha is generate from l2 or mbd data

> **data_path** : ***str*** <br> The template path of the sta to be evaluated. Replace the specific date and skey with "{date}" and "{skey}", respectively.

> **data_source** : ***{local, DFS}*** <br> Whether the data is saved locally or on DFS database.

> **alpha_name**: <br> The list of alpha names for buy / sell side
>> **buy** : ***str*** <br> The column name for the buy-side alpha. When there are multiple alphas in the current data source, use a regular expression to match the column names. Also specify a group named as "label", which will be used to name the different alphas in the evaluation report. For example, "yHatBuy(?P<label>[0-9]*)$" will match columns like "yHatBuy7" and use the number at the end of the colume name as the identifier for different alphas in the same data source. If not familiar with regular expression, you can check this [doc](https://docs.python.org/3/howto/regex.html) and expecially this [part](https://docs.python.org/3/howto/regex.html#non-capturing-and-named-groups) for named groups. <br>
>> **sell** : ***str*** <br> The column name for the sell-side alpha. Can also be a regular expression used to match the column names.

> **pool_name** : ***str*** <br> This applies only if `data_source` is DFS. The pool name where the sta is stored.

> **namespace** : ***str*** <br> This applies only if `data_source` is DFS. The namespace where the sta is stored.

### **eval_focus** : ***{ret, oppo, mixed}***
The evaluation method. Check this for more detailed description.

- "ret" : compare the return with fixed number of opportunities or fixed percentage of opportunities to look at
- "oppo" : compare the number of opportunities with return aligned
- "mixed" : compare the improvement in both number of opportunities and return with a comprehensive method mimicking the production logic

### **target_return** : ***str***
The column name for the target return to be used. The available actual returns on DFS are actualRet90s, actualRet150s, actualRet300s, actualRet600s. For other actual returns, pass a `target_return` with different horizon and set `compute_ret` to True.

### **use_meta** : ***bool***
Whether to use the data in the meta file. This applies only if `eval_focus` is set to "ret". If True, the target number of opportunities per stock per day will be calculated using the data recorded in the meta file. The number in the meta file is closer to the production logic.

### **target_cut** : ***str***
The cutoff method. "top{x}" will target x opportunities while "top{x}p" will target the top x percent opportunities, where x is an integer. "top{x}p" is supported when `eval_focus` is set to "mixed" or "ret" while "top{x}" is only supported when `eval_focus` is set to "ret".

### **compute_ret** : ***bool***
Whether to compute the return using the md data or read from the available actual returns on DFS.

### **lookback_window** : ***int***
The maximum looking back period for factor construction. The unit is second.

### **save_path** : ***str***
The directory to save the temporary evaluation statistics and the evaluation report. It is not necessary to create this path.

### **delete_stats** : ***bool***
Whether to delete the temporary evaluation statistics after generation of the evaluation report.

### **log_path** : ***str***
The directory to save the output and error files when running the evaluation program on HPC. **Please make sure that this path exists on your HPC**.

### **save_summary** : ***bool***
Whether to save the cutoff summary sheet. If True, it will be saved at the same directory as the evaluation report.

### **display** : ***None or list***
The plots or tables to display in the evaluation report. If None, will display the default plots and tables. If a list, will only display those parts assigned in the list. Currently we support:
* all_summary
* all_hist
* cutoff_summary
* cutoff_summary_compact
* group_performance
* group_oppo
* daily_hurdle
* monthly_return
* monthly_oppo
* daily_return
* daily_oppo
* intraday_return
* intraday_oppo
* alpha_decay

### **file_name** : ***None or str***
If not None, will use this as the name for the evaluation report. Otherwise use the default file name. (no need to add the filename extension)

## **Evaluation Methods in Detail**
In the current evaluation system, we filter out independent opportunities so that for each two consecutive opportunities:
* time interval is greater than 1 second
* trading volume is greater than 1500
* trading amount is greater than 15000 

Currently we support three evaluation methods:

### *ret*
Target on fixed number of opportunities and compare the value-weighted realized return on these opportunities. Take 150 opportunities as an example, we will try to pick top *x* percent opportunities so that the number of independent opportunities after filtering is around 150.

### *oppo*
Target on a baseline value-weighted realized return and compare the number of opportunities that achieved this return. The baseline returns vary in different days and we set a minimal baseline return of 2bps. Similarly, we will try to pick top *x* percent opportunities so that the value-weighted realized return is closed to the target baseline each day.

### *mixed*
Compare both the number of opportunities and the value-weighted realized return comprehensively. Under this method, we first filter all the ticks and find out the number of separate ticks. Second, we use, say, 5 percent of the number of separate ticks as the target number of opportunities for this alpha. Third, we try to pick top *x* percent opportunities from the original (**not filtered**) ticks so that the number of independent opportunities is close to the target. This is the reason why we **must set *"top{x}p"* instead of *"top{x}"* under this method**. This method should be used when the alphas are generated from different datasets or using different sample methods.

### **Best practice suggestions**

When comparing alphas generated from the same dataset, you should run *"ret"* evaluation first with `use_meta` set to True and `target_cut` set to top5p (or top{x}p as you wish). In this case you can still run *"oppo"* if you want to know how many more opportunities can the alphas take. 

When comparing alphas generated from different datasets (e.g. l2 and mbd) or different sampling methods (e.g. volume sampling and time sampling), you can first run the *"mixed"* evaluation to see how much improvement there is for the number of opportunities and the realized return. If you want to check the sole improvement on the realized return or the number of opportunities with the other factor controlled, you can use *"ret"* or *"oppo"* evaluation.


## **Intermediate Evaluation Statistics**
We will save the intermediate evaluation statistics locally for the purpose of generating the evaluation report. The procedure is similar to the MapReduce system. You are also free to check these intermediate stats.

### *daily stats*
Statistics per **alpha** per **side** per **stock** per **day**. They include statistics for alphas aggregated over the whole sample and the cutoff part.

### *intraday stats*
Statistics per **alpha** per **side** per **exchange** per **minute** aggregated over the whole sample.


## **Evaluation Report**
Three example evaluation reports corresponding to the three evaluation methods are shown under the folder `./example_results`.

### *summary statistics*
Summary statistics of alphas and the actual return in the whole sample with a histogram plot.

### *overall performance*
* countOppo: average number of opportunities per day per stock
* topPercent: average percentage of number of opportunities across the number of ticks
* yHatHurdle: minimum yHat of the opportunities
* yHatAvg: average yHat of the opportunities
* vwActualRetAvg: notional weighted realized return of each opportunity
* availNtl: average available notional of each opportunity 

### *realized return* / *number of opportunities*
-- Notional weighted realized return of each opportunity aggregated over each minute/day/month.

-- Average number of opportunities per stock per day aggregated over each day/month.

-- Average number of opportunities per stock per minute aggregated over the whole sample

When `eval_focus` is set to *ret*, the report will only display the plots for realized return. If *oppo*, it will only display the plots for the number of opportunities. If *mixed*, plots for both realized return and the number of opportunities will be shown.

### *alpha decay*
A plot of value-weighted realized return aggregated over opporutnities against the time elapsed.