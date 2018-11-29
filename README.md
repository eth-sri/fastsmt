FastSMT  <a href="https://www.sri.inf.ethz.ch/"><img width="100" alt="portfolio_view" align="right" src="http://safeai.ethz.ch/img/sri-logo.svg"></a>
=============================================================================================================

FastSMT is a tool to augment your SMT solver by learning to optimize its performance for your dataset of formulas.
Our tool is built on top of Z3 SMT solver (https://github.com/Z3Prover/z3). Currently we support Z3 4.6.2.

## Setup Instructions

(Optional) Setup python virtual environment. The code is tested with python version 3.5:

```bash
$ git clone git@gitlab.inf.ethz.ch:OU-VECHEV/fastsmt.git # TODO(Mislav): Replace this with the actual link
$ virtualenv -p python3 --system-site-packages venv
$ source venv/bin/activate
$ cd fastsmt
(venv) $ python setup.py install
```

Install and compile Z3 4.6.2 (with Python bindings):

```bash
(venv) $ git clone https://github.com/Z3Prover/z3.git z3
(venv) $ cd z3

# Checkout Z3 version 4.6.2 that we tested against
(venv) $ git checkout 5651d00751a1eb40b94db86f00cb7d3ec9711c4d 

# To generate correct python bindings make sure you activated the virtual env before Z3 compilation
(venv) $ python scripts/mk_make.py --python
(venv) $ cd build
(venv) $ make # (optional) use `make -j4` where 4 is the number of threads used to compile Z3, will likely take couple of minutes
(venv) $ sudo make install
(venv) $ cd ../..
``` 

Finally, compile C++ runner: 
```bash
$ cd fastsmt/cpp
$ make -f make_z3_4.6.2
$ cd ..
```

Please run the tests to make sure installation was successful:

```bash
$ ./test/run_tests.sh
OK!
```

## Creating the dataset

In this section we describe how to prepare the data for synthesis procedure. Our running example will be *leipzig* formulas (QF_NIA theory) from SMT-LIB benchmarks. First, download the formulas into `examples` subdirectory.

```bash
$ mkdir examples
$ cd examples
$ wget http://smt-lib.loria.fr/zip/QF_NIA.zip
$ unzip QF_NIA.zip
$ mkdir QF_NIA/leipzig/all
$ mv QF_NIA/leipzig/*.smt2 QF_NIA/leipzig/all/
$ cd ..
```

You should always create `all` subdirectory in which you should put all of the formulas in SMT2-LIB format.
If you want to split your dataset into training/validation/test set in the ratio 50/20/30 then you use the following command:

```bash
(venv) $ python scripts/py/dataset_create.py --split "50 20 30" --benchmark_dir examples/QF_NIA/leipzig
```

In subfolders `examples/QF_NIA/leipzig/train`, `examples/QF_NIA/leipzig/valid`, `examples/QF_NIA/leipzig/test` you can find formulas which belong to training, validation and test set. 

## Learning and synthesis

Synthesis script performs search over the space of possible strategies and tries to synthesize best strategy for each instance.
It receives configuration for the synthesis procedure, benchmark directory and other information. 

Help menu with information about all possible arguments can be accessed with:

```bash 
(venv) $ python synthesis/learning.py -h
```

Here is an example of the full command (execution should take few minutes):

```bash
(venv) $ python synthesis/learning.py experiments/configs/leipzig/config_apprentice.json \
                --benchmark_dir examples/QF_NIA/leipzig/ \
                --max_timeout 10 \
                --num_iters 5 \
                --iters_inc 5 \
                --pop_size 1 \
                --eval_dir eval/synthesis/ \
                --smt_batch_size 100 \
                --full_pass 2 \
                --num_threads 10 \
                --experiment_name leipzig_example
```

Learned strategies are saved in directory given as `eval_dir`. As a guideline for setting the parameters of the learning procedure, we suggest to look at our experiments in the `experiments` subfolder. In order to synthesize a final strategy from learned strategies use (running the script should take some time):

```bash
(venv) $ python synthesis/multi/multi_synthesis.py \
         --cache cache_leipzig_multi.txt \
         --max_timeout 10 \
         --benchmark_dir examples/QF_NIA/leipzig/ \
         --num_threads 10 \
         --strategy_file output_strategy.txt \
         --leaf_size 4 \
         --num_strategies 5 \
         --input_file eval/synthesis/leipzig_example/train/2/strategies.txt
```

## Additional information

In order to reproduce experimental results from our paper, consult README file in `fastsmt/experiments` subfolder. For more details on the configuration of our system, consult README in `fastsmt/experiments/configs`.

## Tensorboard integration

We use TensorboardX (https://github.com/lanpa/tensorboardX) to provide integration with Tensorboard. In order to inspect the training process of neural network model, you can run Tensorboard server with (where we assume Tensorboard saves data in `runs/` folder):

```bash
(venv) $ tensorboard --logdir runs/
```

You can view the plots at: `http://127.0.0.1:6006` where `6006` is the deafult port to which Tensorboard is logging the data.

## Validation

In order to evaluate final strategy synthesized by our system we provide a validation script. For an input, this script receives dataset with SMT2 formulas and a strategy. It runs Z3 solver with and without using given strategy and outputs the performance comparison (in terms of number of solved formulas and runtime).

```bash
(venv) $ python scripts/py/validate.py \
        --strategy_file output_strategy.txt \
        --benchmark_dir examples/QF_NIA/leipzig/test \
        --max_timeout 10 \
        --batch_size 4
```









