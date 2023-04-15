# Running the Applications

Given the number of applications, options, datasets, and GPU configurations, we have tried to simplify application testing as much as possible. To facilitate test sweeps across 1 to 16 GPUs, multiple application options, and datasets, every application has two associated scripts: `hive-mgpu-run.sh` and `hive-application-test.sh`.

In general `hive-mgpu-run.sh` deals with parameter sweeps and schedules `hive-application-test.sh` as multi-GPU SLURM jobs. The `hive-application-test.sh` script generally deals with datasets and associated paths, and configures itself with the parameters necessary to run the application.

### Default Run Configuration
The simplest way to run an application is to execute:

```
./hive-mgpu-run.sh
```
The associated `hive-application-test.sh` will execute with datasets in the following user directories on NVIDIA's `nslb` cluster:

```
/home/u00u7u37rw7AjJoA4e357/data/gunrock/gunrock_dataset
/home/u00u7u37rw7AjJoA4e357/data/gunrock/hive_datasets
```

### Alternate Run Configurations
Additional command line parameters and / or script modifications are necessary to run on additional datasets or with alternate application parameters.

#### hive-mgpu-run.sh

This script configures SLURM with `NUM_GPUS` to sweep across on a chosen `PARTITION_NAME`. Running the script with no parameters (as shown above) is equivalent to: 

```
./hive-mgpu-run.sh 16 dgx2 
```
This runs `hive-application-test.sh` across 1 to 16 GPUs on the machine partition named `dgx2`.

For some applications, this script might have additional parameter variables that are worth exploring and modifying. Please see the individual HIVE application chapters for more details.

#### hive-application-test.sh

The primary reason to modify this script is to provide additional dataset information. In general these scripts will include some or all of the following arrays: 
	
* `DATA_PREFIX` path to directory containing desired dataset
* `NAME` a simple string naming the dataset, generally sans a file extension (e.g., `NAME[0]="twitter"` for `twitter.mtx`) 
* `GRAPH` aggregated options for the chosen dataset to pass to the application (i.e., combine `DATA_PREFIX` and `NAME` with additional information expected by the application)

**Please note** that you must update the associated for loop index if you add or remove items to the arrays mentioned.


### Future Script Simplification

In the future we would like to refactor `hive-mgpu-run.sh` to simply configure the necessary SLURM command (e.g., resources and hardware partition) and pass the command to the `hive-application-test.sh` script. The application script can then deal with sweeping across its relevant parameters and datasets.
