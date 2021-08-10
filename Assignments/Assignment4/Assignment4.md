# Connecting to Campus Cluster
The most common ways to login is to use secure shell (SSH) as,
```shell
$ ssh -l <net-id-without-at-illinois> cc-login.campuscluster.illinois.edu
```
This should work on all Linux/Mac. If it does not work, explore other [clients](https://campuscluster.illinois.edu/resources/docs/user-guide/#connect) in the campus cluster wiki.

	Note: It is easier to use a UNIX-based system because of similarity in syntax

Upon login to the campus cluster, you will be able to see your quota - space in `home` and `scratch` etc. Sometimes your job may create significant output. In case your `home` area fills up, you cannot create files. A good practice is to set the results directory in `scratch` and then transfer to `home`. Be careful though since `scratch` is cleared periodically, and there is no _undo_. Run
```shell
$ quota
```
periodically to check the status of your `home` and `scratch`.

# Environment and modules
There are pre-packaged environment configured made avaialable for specific applications. For example, we may want to use `anaconda` to create virtual environments, `git` to version control code. We can check the available modules using
```shell
$ module avail
```
and loaded modules using
```
$ module list
```
Some modules will be used more frequently that others - like `git`, `vim`, `anaconda`. So it is convenient to add those to your `.bashrc` file. An example is the following line appended to your `.bashrc`.
```bash
module load vim curl git anaconda/3
```
Create a backup of the original `.bashrc` before modifying this file though, for example,
```
$ cp .bashrc .bashrc.bak
```
Once you have loaded `git`, you can clone the `bilby` repository and create a conda environments where you install it. Create a developemental branch and add some `log.info` to various places in the source code of bilby to see what is going on inside.

# Slurm directives
The slurm commands you will use to submit and monitor jobs are `sbatch`, `scancel`, `squeue`, `sinfo`, and `scontrol`. First two are to submit and cancel jobs, the remaining are to monitor the status of the queue, information on nodes, or to check the features of the node you want to submit to. For example,
```
$ sinfo -p GravityTheory
```
tells you there are two node names, and their state.
```
$ scontrol show node ccc0216
```
gives some details, like there are 128 CPUs in GravityTheory nodes. For other subcommands check out the [slurm documentation](https://slurm.schedmd.com/documentation.html).

# Submission
Submission is done using the `sbatch` command, usually as,
```
$ sbatch <filename-containing-directives>
```

Treat your submit file as any other file with some `#SBATCH ...` lines added that tells slurm what to do. For example, the following is a python file with some `SBATCH` directives.
```python
#!/usr/bin/env python
#SBATCH --partition=GravityTheory
#SBATCH --time=00:03:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=1G
#SBATCH --output=test.out
#SBATCH --error=test.err

import multiprocessing
import os
import time


def mul(a_b):
    time.sleep(5)
    a, b = a_b
    result = a * b
    print(
        '%s says that %s%s = %s' % (
            multiprocessing.current_process().name,
            'mul', a_b, result
        ), flush=True
    )


def plus(a_b):
    time.sleep(5)
    a, b = a_b
    result = a + b
    print(
        '%s says that %s%s = %s' % (
            multiprocessing.current_process().name,
            'plus', a_b, result
        ), flush=True
    )


if __name__ == '__main__':
    with multiprocessing.Pool(4) as pool:
        pool.map(
            mul, zip(
                range(1, 100), range(11, 110)
            )
        )
```
As you submit the job, the `print` statements will be redirected to the `test.out` file. Monitor it continuously using
```bash
$ tail -f test.out
```
If there is an error it will go to `test.err`.

Note that the output comes in 4 at a time. Try a few different combinations
- Keep `cpus-per-task=4` but change the pool size to 8, and vice versa
- Remove the `time.sleep` with some CPU bound tasks (actual computation - compute squares, prime factorization etc.). Report any difference you observe.
- Change the number of `nodes` to 2. Observe any differences?
- Do an `squeue -u <your-username>` as the job is running. While _it is still running_ you can `ssh` into one of the compute nodes, like,
```bash
$ ssh ccc0216
```
Check what all is running once you are in the compute node, for example
```bash
$ top -u <your-username>
```
Observe what you find when you set `--nodes=2`. Why is this the case?

# Usual use case
Usually, you will submit jobs via a shell script wrapper. For example, (check out the other `SBATCH` directives)
```bash
#!/bin/bash -l
#SBATCH --partition=GravityTheory
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=1G
#SBATCH --output=bilby.out
#SBATCH --error=bilby.err

# activate your environment, for example,
conda activate my-env-to-run-bilby

# you can change working directories if needed
python my-script-that-runs-a-bliby-job
```
In case you have some other executable, replace the last line with it and the command line arguments.

# Few Exercises
- Convert your bilby python script to an executable, add command line arguments like the pool size, all injection parameters in case of doing an injection etc. Read about the `argparse` library in case you are unfamiliar. Look at the source code of `bilby_pipe`.
- In the python script above, the argument of the functions `mul` and `plus` take in a tuple. Change the function signature to the usual,
```python
def plus(a, b):
```
Now write a decorator so that this signature can directly be used with `multiprocessing.map` as used here.
