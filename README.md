FastSMT  <a href="https://www.sri.inf.ethz.ch/"><img width="100" alt="portfolio_view" align="right" src="http://safeai.ethz.ch/img/sri-logo.svg"></a>
=============================================================================================================

<a href="http://fastsmt.ethz.ch/"><img width="700" alt="portfolio_view" align="center" src="http://fastsmt.ethz.ch/img/cover.png">


- [ ] TODO: Write short description of the project, link to the paper, etc.

## Setup Instructions

(Optional) Setup python virtual environment. The code is tested with python version 3.5:

```bash
$ git clone git@gitlab.inf.ethz.ch:OU-VECHEV/fastsmt.git # TODO(Mislav): Replace this with the actual link
$ virtualenv -p python3 --system-site-packages venv
$ source venv/bin/activate
(venv) $ python setup.py install
```

Our tool is built on top of Z3 SMT solver (https://github.com/Z3Prover/z3).
All our experiments were ran using `Z3 4.6.2` which was latest version of Z3 solver at the time. 
We also support `Z3 4.8.4` although it was not tested as thoroughly as `Z3 4.6.2` (note that there are few differences in setup).
However, we only guarantee reproducibility of results for experiments in 
the subdirectory `experiments` for the older version (we did not test with `4.8.4`)

```bash
$ git clone https://github.com/Z3Prover/z3.git z3
$ cd z3

# (Optional) checkout Z3 version 4.6.2 that we tested against
# $ git checkout 5651d00751a1eb40b94db86f00cb7d3ec9711c4d 

# To generate correct python bindings you need to activate the virtual env before Z3 compilation
$ source ../venv/bin/activate
(venv) $ python scripts/mk_make.py --python
(venv) $ cd build
(venv) $ make # (optional) use `make -j4` where 4 is the number of threads used to compile Z3, will likely take couple of minutes
(venv) $ make install
(venv) $ cd ../..
``` 

Finally, compile C++ runner: 
```bash
$ cd fastsmt/cpp
$ make -f make_z3_4.8.4 # make -f make_z3_4.6.2 if you built Z3 4.6.2
$ cd ..
```

Optionally, compile and install FastText. This is only needed if you want to run with bilinear model. Then install python bindings and add the folder to PYTHONPATH. 
This will create the FastText binary and also all relevant libraries (shared, static, PIC):

```bash
$ git clone https://github.com/facebookresearch/fastText.git
$ cd fastText
$ git checkout a5d22aba45f38c12d195ecc6c3e448aa3690fbbd
$ mkdir build && cd build && cmake ..
$ make 
$ cd ..
# FastText bindings for Python
$ pip install .
$ cd ..
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
(venv) $ python synthesis/synthesis.py -h
```

Here is an example of the full command:

```bash
(venv) $ python synthesis/synthesis.py experiments/configs/leipzig/config_apprentice.json \
                --benchmark_dir examples/QF_NIA/leipzig/ \
                --max_timeout 10 \
                --num_iters 10 \
                --iters_inc 10 \
                --pop_size 1 \
                --eval_dir eval/synthesis/ \
                --smt_batch_size 100 \
                --full_pass 10 \
                --num_threads 30 \
                --experiment_name leipzig
```

Results of the synthesis are saved in `eval_dir` which contains synthesized strategies in each of the passes, for both training and validation dataset.

In order to combine the synthesized strategies into a final strategy in SMT2 format use:

```bash
(venv) $ python synthesis/multi/multi_synthesis.py \
         --cache cache/leipzig_multi.txt \
         --max_timeout 10 \
         --benchmark_dir examples/QF_NIA/leipzig/ \
         --num_threads 10 \
         --strategy_file output_strategy.txt \
         --leaf_size 100 \
         --num_strategies 5 \
         --input_file eval/synthesis/leipzig/train/10/strategies.txt
```

For the full list of hyperaparameters and their meaning please consult our paper.

## Configuration

In this section we describe how to configure our system using JSON configuraton file.

### Main and exploration model

First, you need to give some general information on `main model` and `exploration model` which are going to be used during the search. When choosing a tactic, exploration model is selected with some (decaying) probability specified according to `exploration` field in the configuration. In this field you specify the initial exploration probability, decay of exploration probability and minimum exploration probability (when this is reached, decay stops).
Next, you specify `pop_size` which is number of strategies that should be evaluated in every iteration (for a single formula instance). 

```json
    "main_model": "apprentice",
    "explore_model": "random",
    "exploration": {
	"enabled": true,
        "init_explore_rate": 1,
        "explore_decay": 0.99,
        "min_explore_rate": 0.05
    },
    "pop_size": 10,
```

### Model configuration

Next, you need to specify parameters of the model. Here we describe parameters of the apprentice model.

* `min_train_data` - smallest number of data samples needed to train the model
* `type` - feature type to use during the training, for best results use `bow`
* `tactic_embed_size` - dimension of the embedding of each tactic
* `adam_lr` - learning rate of Adam optimizer used during training
* `epochs` - number of epochs used during the training if early stopping is not used
* `mini_batch_size` - size of the mini batch during the training
* `early_stopping_inc` - stop early if validation error increases `early_stopping_inc` times in a row
* `valid_split` - split collected data in train/valid in this ratio (e.g. if `valid_split` = 0.8 then 80% of data will be used for training)
* `min_valid_samples` - minimum number of samples needed to use early stopping (otherwise use fixed number of epochs)

```json
    "models": {
        "apprentice": {
            "min_train_data": 50,
            "type": "bow",
            "tactic_embed_size": 30,
            "adam_lr": 0.0001,
            "epochs": 100,
            "mini_batch_size": 64,
            "early_stopping_inc": 3,
            "valid_split": 0.8,
            "min_valid_samples": 400
        }
    },
```

### Tactics configuration

Finally, you need to specify tactics over which model will perform its search. First, in field `all_tactics` you should list names of all tactics which you want to use. Next, you should specify parameters which model can set for each tactic. If you do not specify a parameter its default value will be used. We distinguish between `boolean` and `integer` parameters. For `boolean` parameters it is enough to list them for each tactic. For example, tactic `simplify` has specified boolean parameters `elim_and`, `som`, `blast_distinct`, etc. For integer parameters you should provide lower and upper bound (inclusive) on the values you want to search.

```json
    "tactics_config": {
        "all_tactics": [
            "simplify",
            "smt",
            "bit-blast",
            "propagate-values",
            "ctx-simplify",
            "elim-uncnstr",
            "solve-eqs",
            "lia2card",
            "max-bv-sharing",
            "nla2bv",
            "qfnra-nlsat",
            "cofactor-term-ite"
	    ],
        "allowed_params": {
            "simplify": {
                "boolean": [
                    "elim_and",
                    "som",
                    "blast_distinct",
                    "flat",
                    "hi_div0",
                    "local_ctx",
                    "hoist_mul"
                ]
            },
            "nla2bv": {
                "integer": [
                    ["nla2bv_max_bv_size", 0, 100]
                ]
            }
        }
    }

```

## Tensorboard integration

We use TensorboardX (https://github.com/lanpa/tensorboardX) to provide integration with Tensorboard. In order to inspect the training process of neural network model, you can run Tensorboard server with (where we assume Tensorboard saves data in `runs/` folder):

```bash
(venv) $ tensorboard --logdir runs/
```

You can view the plots at: `http://127.0.0.1:6006` where `6006` is the deafult port to which Tensorboard is logging the data.

## Validation

In order to evaluate final strategy synthesized by our system we provide a validation script. For an input, this script receives dataset with SMT2 formulas and a strategy. It runs Z3 solver with and without using givem strategy and outputs the performance comparison.

```bash
(venv) $ python scripts/py/validate.py \
        --strategy_file output_strategy.txt \
        --benchmark_dir examples/QF_NIA/leipzig/test \
        --max_timeout 10 \
        --batch_size 4
```





