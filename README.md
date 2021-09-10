# **Table of contents**
1. [Usage](#usage)
2. [Configuration Parameter](#configuration-parameter)
3. [Evaluation Method](#evaluation-method)
4. [Intermediate Evaluation Statistics](#intermediate-evaluation-statistics)
5. [Evaluation Report](#evaluation-report)

## **Usage**
1. Clone into this repo! Use the following ssh otherwise it's slow. 

    ``git clone git@github.com:study-int/STA_evaluation.git``
2. Modify the .yaml configuration file
3. Run from the command line
    
    ``python eval_run.py <configuration file path>``

4. Check the evaluation report saved in the assigned `save_path` in the configuration file

### Prerequisite
* update your dfs api to jdfs >= 1.1.0. You can set up the dfs following the instruction in this [link](https://github.com/rz475743/ceph_client).
* make sure you can use git via ssh keys


### Example
``python eval_run.py sta_input_ps_example.yaml``

## **Configuration Parameter**:
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
>> **buy** : ***str*** <br> The column name for the buy-side alpha. When there are multiple alphas in the current data source, use a regular expression to match the column names. Also specify a group named as "label", which will be used to name the different alphas in the evaluation report. For example, "yHatBuy(?P<label>[0-9]*)$" will use the number at the end of the colume name as the identifier for different alphas in the same data source. If not familiar with regular expression, you can check this [doc](https://docs.python.org/3/howto/regex.html) and expecially this [part](https://docs.python.org/3/howto/regex.html#non-capturing-and-named-groups) for named groups. <br>
>> **sell** : ***str*** <br> The column name for the sell-side alpha. Can also be a regular expression used to match the column names.

> **pool_name** : ***str*** <br> This applies only if `data_source` is DFS. The pool name where the sta is stored.

> **namespace** : ***str*** <br> This applies only if `data_source` is DFS. The namespace where the sta is stored.

### **eval_focus** : ***{ret, oppo, mixed}***
The evaluation method. Check this for more detailed description.
    
- "ret" : compare the return with fixed number of opportunities to look at (for example 240)
- "oppo" : compare the number of opportunities with return aligned
- "mixed" : compare the improvement in both number of opportunities and return with a comprehensive method mimicking the production logic

### **target_return** : ***str***
The column name for the target return to be used. The available actual returns on DFS are actualRet90s, actualRet150s, actualRet300s, actualRet600s. For other actual returns, pass a `target_return` with different horizon and set `compute_ret` to True.

### **target_cut** : ***str***
The cutoff method. "top{x}" will target x opportunities while "top{x}p" will target the top x percent opportunities, where x is an integer. "top{x}p" is supported when `eval_focus` is set to "mixed" while "top{x}" is supported when `eval_focus` is set to "ret".

### **compute_ret** : ***bool***
Whether to compute the return using the md data or read from the available actual returns on DFS.

### **lookback_window** : ***int***
The maximum looking back period for factor construction. The unit is second.

### **save_path** : ***str***
The directory to save the temporary evaluation statistics and the evaluation report.

### **log_path** : ***str***
The directory to save the output and error files when running the evaluation program on HPC. **Please make sure that this math exists on your HPC**.


## **Evaluation Method**
In the current evaluation system, we filter out independent opportunities so that for each two consecutive opportunities:
* time interval is greater than 1 second
* trading volume is greater than 1500
* trading amount is greater than 15000 

Currently we support three evluation methods:

### *ret*
Target on fixed number of opportunities and compare the value-weighted realized return on these opportunities. Take 240 opportunities as an example, we will try to pick top *x* percent opportunities so that the number of independent opportunities after filtering is around 240.

### *oppo*
Target on a baseline value-weighted realized return and compare the number of opportunities that achieved this return. The baseline returns vary in different days and we set a minimal baseline return of 2bps. Similarly, we will try to pick top *x* percent opportunities so that the value-weighted realized return is closed to the target baseline each day.

### *mixed*
Compare both the number of opportunities and the value-weighted realized return comprehensively. This will be the most frequently-used evaluation method. 

Under this method, we first filter all the ticks and find out the number of separate ticks. Second, we use, say, 5 percent of the number of separate ticks as the target number of opportunities for this alpha. It means that alpha generated using the same md, for example l2, will have close number of target opportunities. Third, we try to pick top *x* percent opportunities from the original (**not filtered**) ticks so that the number of independent opportunities is close to the target.


## **Intermediate Evaluation Statistics**
We will save the intermediate evaluation statistics locally for the purpose of generating the evaluation report. The procedure is similar to the MapReduce system. You are also free to check these intermediate stats.

### *daily stats*
Statistics per **alpha** per **side** per **stock** per **day**. They include statistics for alphas aggregated over the whole sample and the cutoff (eg. top240) part.

### *intraday stats*
Statistics per **alpha** per **side** per **exchange** per **minute** aggregated over the whole sample.


## **Evaluation Report**
Three example evaluation reports corresponding to the three evaluation methods are under the folder `./results`.

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