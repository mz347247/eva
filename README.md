## **Usage**
1. Modify the .yaml configuration file
2. Run from the command line
    
    ``python eval_run.py <configuration file path>``

3. Check the evaluation report saved in the assigned `save_path` in the configuration file

## **Configuration Parameter**:
 
### **machine** :  ***{personal-server, HPC}***
The machine to run the evaluation program

### **bench** : ***{IC, IF}***
The bench index to read the md and return data

### **universe** : ***{custom, IC, IF}***
The set of stocks to run the evaluation program on. If custom, will use all the stocks under the specified directory

### **njobs** : ***int***
The number of processors on personal server or the number of jobs on HPC used to run the evaluation program

### **start_date** : ***int***
Start date of the evaluation period

### **end_date** : ***int***
End date of the evaluatin period

### **eval_alpha** :
A list of data sources. Each source can contain multiple alphas or just one alpha.
    
> **name** : ***str*** <br> The name of this data source. Will be used to distinguish alphas from different data sources and also shown in the evaluation report.

> **data_type** : ***{l2, mbd}*** <br> Whether the alpha is generate from l2 or mbd data

> **data_path** : ***str*** <br> The template path of the sta to be evaluated. Replace the specific date and skey with "{date}" and "{skey}", respectively

> **data_source** : ***{local, DFS}*** <br> Whether the data is saved locally or on DFS database

> **alpha_name**: <br> The list of alpha names for buy / sell side
>> **buy** : ***str*** <br> The column name for the buy-side alpha. When there are multiple alphas in the current data source, use a regular expression to match the column names. Also specify a group named as "label", which will be used to name the different alphas in the evaluation report. For example, "yHatBuy(?P<label>[0-9]*)$" will use the number at the end of the colume name as the identifier for different alphas in the same data source. If not familiar with regular expression, you can check this [doc](https://docs.python.org/3/howto/regex.html) and expecially this [part](https://docs.python.org/3/howto/regex.html#non-capturing-and-named-groups) for named groups <br>
>> **sell** : ***str*** <br> The column name for the sell-side alpha. Can also be a regular expression used to match the column names

> **pool_name** : ***str*** <br> This applies only if `data_source` is DFS. The pool name where the sta is stored

> **namespace** : ***str*** <br> This applies only if `data_source` is DFS. The namespace where the sta is stored

### **eval_focus** : ***{ret, oppo, mixed}***
The evaluation method
    
- "ret" : compare the return with fixed number of opportunities to look at (for example 240)
- "oppo" : compare the number of opportunities with return aligned
- "mixed" : compare the improvement in both number of opportunities and return with a comprehensive method mimicking the production logic

### **target_return** : ***str***
The column name for the target return to be used. The available actual returns on DFS are actualRet90s, actualRet150s, actualRet300s, actualRet600s. For other actual returns, pass a `target_return` with different horizon and set `compute_ret` to True.

### **target_cut** : ***str***
The cutoff method. "top{x}" will target x opportunities while "top{x}p" will target the top x percent opportunities, where x is an integer. Only "top{x}p" is supported when `eval_focus` is set to "mixed"

### **compute_ret** : ***bool***
Whether to compute the return using the md data or read from the available actual returns on DFS

### **lookback_window** : ***int***
The maximum looking back period for factor construction. The unit is second

### **save_path** : ***str***
The directory to save the temporary evaluation statistics and the evaluation report

### **log_path** : ***str***
The directory to save the output and error files when running the evaluation program on HPC