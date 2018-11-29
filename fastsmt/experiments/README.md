# Intro

Here we describe how to reproduce results reported in our paper. Note that you need to use Z3 version `4.6.2` to get the same results. Please consult setup instructions on this.

# Downloading benchmarks

We provide scripts to download the benchmarks used in our experiments either from our repository or official SMT-LIB repository.
In both cases, after the download relevant datasets can be found in corresponding folders - e.g. `experiments/data/sage2/train/` will contain training set to be used for synthesis on *Sage2* benchmark. 

#### SRI repository (recommended)

```bash
$ ./download_data.sh
```

#### SMT-LIB repository

You can also download our benchmarks directly from the official SMT-LIB repository.
In the case you download benchmarks from the official repository note that they might be different from ours as they are subject to frequent updates from SMT-LIB community. 

```bash
$ ./download_data_smtlib.sh
```

Note that file sizes are much bigger in this case as it also downloads benchmarks not used in our experiments.

# Learning and synthesis

You can find strategies resulting from our experiments in `final_strategies` subdirectory. Concretely, in `all` subdirectory you can find all learned strategies and in `smt2` subdirectory you can find final synthesized strategies in SMT2 format. Next, we describe how to re-run these experiments.

## Learning 

Note that you need FastText set up properly in order to run bilinear model. Please run the following command which will create the FastText binary and also all relevant libraries (shared, static, PIC):

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

After that, add FastText to PYTHONPATH.
In order to train all models and reproduce our experiment on Sage2 benchmark use the following command:

```bash
(venv) $ ./experiments/runners/run_sage2.sh
```

This will sequentially run all models (neural network, bilinear, evolutionary, random and bfs model) used in our experiments. For other benchmarks use appropriate scripts (e.g. ```./experiments/runners/run_leipzig.sh ```). 

## Synthesis

In order to synthesize single master strategy out of all created strategies run (e.g. for Sage2):

```bash
(venv) $ ./experiments/synthesis/tree_sage2.sh
```

# Evaluation

To compare all strategies synthesized by the neural network apprentice model against Z3 with time limit 10 seconds (and reproduce Table 2 in our work) use:
```bash
(venv) $ ./experiments/evaluate_all_vs_z3.sh
```

To compare all strategies synthesized by the neural network apprentice model against Z3 with time limit 10 minutes (and reproduce Table 5 in our work) use:
```bash
(venv) $ ./experiments/evaluate_all_vs_z3_10min.sh
```

To compare final strategies after synthesis with Z3 (and reproduce Table 3 in our work) use:

```bash
(venv) $ ./experiments/evaluate_final_against_z3.sh
```

In order to compare search models and reproduce Figures 3 and 4 from our work use:

```bash 
(venv) $ ./experiments/runners/run_sage2_test.sh
(venv) $ python scripts/py/analyze_synthesis.py --eval_dir experiments/eval/sage2/ --models sage2_apprentice:10 sage2_random:1 sage2_bilinear:10 sage2_bfs:1 sage2_evo:1 --folder valid --legend
(venv) $ python scripts/py/analyze_synthesis.py --eval_dir experiments/eval/sage2/ --models sage2_apprentice:10 sage2_apprentice:8 sage2_apprentice:6 sage2_apprentice:4 sage2_apprentice:2 --folder valid --legend
```

