# Leipzig
python experiments/eval_vs_z3.py --benchmark_dir experiments/data/qf_nia/leipzig/test/ --strategies_file experiments/eval/leipzig/leipzig_apprentice/train/10/strategies.txt --max_timeout 600 --formulas_batch 10 --strategies_batch 11

# Sage2
python -u experiments/eval_vs_z3.py --benchmark_dir experiments/data/sage2/small_test/ --strategies_file experiments/eval/sage2/sage2_apprentice/train/10/strategies.txt --max_timeout 600 --formulas_batch 1 --strategies_batch 60 --max_formulas 800

# hycomp
python experiments/eval_vs_z3.py --benchmark_dir experiments/data/qf_nra/hycomp/small_test/ --strategies_file experiments/eval/hycomp/hycomp_apprentice/train/10/strategies.txt --max_timeout 600 --formulas_batch 1 --strategies_batch 120 --max_formulas 325 

# core
python experiments/eval_vs_z3.py --benchmark_dir experiments/data/qf_bv/bruttomesso/core/test/ --strategies_file experiments/eval/core/core_apprentice/train/10/strategies.txt --max_timeout 600 --batch_size 100

# AProVE
python experiments/eval_vs_z3.py --benchmark_dir experiments/data/qf_nia/AProVE/small_test/ --strategies_file experiments/eval/aprove/aprove_apprentice/train/10/strategies.txt --max_timeout 600 --formulas_batch 1 --strategies_batch 120 --max_formulas 315 
