# Assignment 0: Jupyter Notebook usage and assignment submission workflow

Hi all! Welcome to the IERG 5350 RL course!

This assignment is used to illustrate how to use Jupyter Notebook and how to submit your assignment to Blackboard!




## Environment setup instruction

In this course, we require you to have the basic knowledge of python.
 
In each assignment, some useful packages will be used to help you. 
For example the reinforcement learning environment [Gym](https://gym.openai.com/), scientific computing [Numpy](https://numpy.org/), machine learning framework [PyTorch](https://pytorch.org/) etc.
We will list the packages required at each assignment. 
Therefore in the very beginning of setting up the environment, 
you only need to set up your python environment. 

We highly recommend you to setup a conda virtual environment where to finish all assignments.
Here is the advantages to do so:

1. The packages installed during this course will not affect other projects on your computers since the environment is independent of other projects.
2. Other members can run your codes in this course seamless. Since we all using the same environment and packages.
3. The robustness and compatibility of codes is also an important criterion to assess your completion of assignments. This is because if the program is not runnable at TA's computer, your code is considered as not runnable. So, you know.
4. In your future research career, a clear and ordered code management habit is one of the key to success.

We recommend you to use anaconda python environments. First, download the package and install anaconda following the instruction at https://www.anaconda.com/distribution/

Then create your environment via typing command line:

```
conda create -n ierg5350 python=3.7
```

By doing this, you created an environment name `IERG5350` with python 3.7 installed. 
Then you need to activate your environment before doing anything:

```
conda activate ierg5350
```

If you activate the environment successfully, you will see `(ierg5350) COMPUTERNAME:~ USERNAME$` at your shell.

Then you can install the packages we listed at each assignment like:

```
pip install XXX==1.0.0
```

where the `XXX==1.0.0` means to install package `XXX` with the specified version `1.0.0`. The packages' names and versions will be listed at each assignment.

If you use other packages that you think helpful, you need to list them with the version number at your report. Make sure the extra package DO NOT help you to finish the essential part of the assignment. The following example is NOT acceptable.

```python
import numpy as np
from kl_divergence_package_writte_by_smart_guys import get_kl

def compute_kl(dist1, dist2):
    """
    Problem 1: You need to implement the computing of KL
    Divergence given two distribution instances.
    
    You should only use numpy package.
    
    The return should be a float that greater than 0.
    """
    return get_kl(dist1, dist2)
```


## Install and use  jupyter notebook

In some assignments, we only provide you with a single jupyter notebook file. 
To open and edit the notebook, you have to install the package first as follows:

```
conda activate ierg5350
pip install notebook
```

Now you have installed the jupyter notebook. Go to the directory such as `assignment0`, command:

```
jupyter notebook
```

Now you have opened a jupyter notebook session in your computer. 
Open your browser and go to `http://localhost:8888`  (8888 is the port number, you can change it by starting jupyter notebook via `jupyter notebook --port 8889`).
You will see a beautiful interface provided by jupyter notebook. Now click into `FILE.ipynb` and start coding!

For more information, please visit: https://jupyter.org/install.html

Now, please go through the `assignment0.ipynb`.



------

*2021-2022 1st term, IERG 5350: Reinforcement Learning. Department of Information Engineering, The Chinese University of Hong Kong. Course Instructor: Professor ZHOU Bolei. Assignment author: PENG Zhenghao.*