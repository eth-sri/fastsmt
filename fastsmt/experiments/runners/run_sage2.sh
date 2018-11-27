# BFS
python synthesis/synthesis.py experiments/configs/sage2/config_bfs.json --benchmark_dir experiments/data/sage2/ --max_timeout 10 --num_iters 100 --eval_dir experiments/eval/sage2/ --smt_batch_size 60 --full_pass 1 --num_threads 4 --experiment_name sage2_bfs --pop_size 1 

# Random
python synthesis/synthesis.py experiments/configs/sage2/config_random.json --benchmark_dir experiments/data/sage2/ --max_timeout 10 --num_iters 100 --eval_dir experiments/eval/sage2/ --smt_batch_size 230 --full_pass 1 --num_threads 80 --experiment_name sage2_random2 --pop_size 1

# NN
python synthesis/synthesis.py experiments/configs/sage2/config_apprentice.json --benchmark_dir experiments/data/sage2/ --max_timeout 10 --num_iters 10 --iters_inc 10 --eval_dir experiments/eval/sage2/ --smt_batch_size 230 --full_pass 10 --num_threads 40 --experiment_name sage2_apprentice --pop_size 1

# FastText
python synthesis/synthesis.py experiments/configs/sage2/config_fast_text.json --benchmark_dir experiments/data/sage2/ --max_timeout 10 --num_iters 10 --iters_inc 10 --eval_dir experiments/eval/sage2/ --smt_batch_size 230 --full_pass 10 --num_threads 80 --experiment_name sage2_bilinear --pop_size 1

# Evolutionary
python synthesis/synthesis.py experiments/configs/sage2/config_evo.json --benchmark_dir experiments/data/sage2/ --max_timeout 10 --num_iters 100 --iters_inc 10 --eval_dir experiments/eval/sage2/ --smt_batch_size 230 --full_pass 1 --num_threads 30 --experiment_name sage2_evo --pop_size 1 --evo
