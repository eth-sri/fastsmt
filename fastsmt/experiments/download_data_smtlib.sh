# NOTE: Below is way to fetch the dataset from official SMT-LIB repository.
#       Beware that download will take much longer as it is downloading more benchmarks than
#       are used in our experiments.

mkdir data
mkdir data/sage2 data/qf_nia data/qf_nra data/qf_bv
git clone https://clc-gitlab.cs.uiowa.edu:2443/SMT-LIB-benchmarks/Sage2.git data/sage2
git clone https://clc-gitlab.cs.uiowa.edu:2443/SMT-LIB-benchmarks/QF_NIA.git data/qf_nia
git clone https://clc-gitlab.cs.uiowa.edu:2443/SMT-LIB-benchmarks/QF_NRA.git data/qf_nra
git clone https://clc-gitlab.cs.uiowa.edu:2443/SMT-LIB-benchmarks/QF_BV.git data/qf_bv

# Leipzig
mkdir -p data/qf_nia/leipzig/all
mv data/qf_nia/leipzig/*.smt2 data/qf_nia/leipzig/all/
python ../scripts/py/dataset_create.py --split "40 20 40" --benchmark_dir data/qf_nia/leipzig/

# Sage2
mkdir -p data/sage2/all
mv data/sage2/*.smt2 data/sage2/all/
python ../scripts/py/dataset_create.py --split "3 10 87" --benchmark_dir data/sage2/

# hycomp
mkdir -p data/qf_nra/hycomp/all
mv data/qf_nra/hycomp/*.smt2 data/qf_nra/hycomp/all
python ../scripts/py/dataset_create.py --split "8 20 72" --benchmark_dir data/qf_nra/hycomp/

# core
mkdir -p data/qf_bv/bruttomesso/core/all
mv data/qf_bv/bruttomesso/core/*.smt2 data/qf_bv/bruttomesso/core/all
python ../scripts/py/dataset_create.py --split "40 20 40" --benchmark_dir data/qf_bv/bruttomesso/core/

# AProVE
mkdir -p data/qf_nia/AProVE/all
mv data/qf_nia/AProVE/*.smt2 data/qf_nia/AProVE/all/
python ../scripts/py/dataset_create.py --split "9 20 71" --benchmark_dir data/qf_nia/AProVE/







