# Leipzig
python scripts/py/validate.py --strategy_file experiments/final_strategies/leipzig_mul.txt --benchmark_dir experiments/data/qf_nia/leipzig/test/ --max_timeout 10 --batch_size 4

# Sage2
python scripts/py/validate.py --strategy_file experiments/final_strategies/sage2_mul.txt --benchmark_dir experiments/data/sage2/test/ --max_timeout 10 --batch_size 4

# core
python scripts/py/validate.py --strategy_file experiments/final_strategies/core_mul.txt --benchmark_dir experiments/data/qf_bv/bruttomesso/core/test/ --max_timeout 10 --batch_size 4

# hycomp
python scripts/py/validate.py --strategy_file experiments/final_strategies/hycomp_mul.txt --benchmark_dir experiments/data/qf_nra/hycomp/test/ --max_timeout 10 --batch_size 4

# AProVE
python scripts/py/validate.py --strategy_file experiments/final_strategies/aprove_mul.txt --benchmark_dir experiments/data/qf_nia/AProVE/test/ --max_timeout 10 --batch_size 4
