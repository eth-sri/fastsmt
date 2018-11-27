# Leipzig
python experiments/eval_vs_z3.py --benchmark_dir experiments/data/qf_nia/leipzig/test/ --strategies_file experiments/eval/leipzig/leipzig_apprentice/train/10/strategies.txt --max_timeout 10 --formulas_batch 20 --strategies_batch 10

# Sage2
python experiments/eval_vs_z3.py --benchmark_dir experiments/data/sage2/test/ --strategies_file experiments/eval/sage2/sage2_apprentice/train/10/strategies.txt --max_timeout 10 --formulas_batch 50 --strategies_batch 20

# hycomp
python experiments/eval_vs_z3.py --benchmark_dir experiments/data/qf_nra/hycomp/test/ --strategies_file experiments/eval/hycomp/hycomp_apprentice/train/10/strategies.txt --max_timeout 10 --formulas_batch 5 --strategies_batch 10

# core
python experiments/eval_vs_z3.py --benchmark_dir experiments/data/qf_bv/bruttomesso/core/test/ --strategies_file experiments/eval/core/core_apprentice/train/10/strategies.txt --max_timeout 10 --formulas_batch 20 --strategies_batch 20

# AProVE
python experiments/eval_vs_z3.py --benchmark_dir experiments/data/qf_nia/AProVE/test/ --strategies_file experiments/eval/aprove/aprove_apprentice/train/10/strategies.txt --max_timeout 10 --formulas_batch 5  --strategies_batch 20
