# BFS
python synthesis/learning.py experiments/configs/sage2/config_bfs.json --smt_batch_size 100 --benchmark_dir experiments/data/sage2/ --max_timeout 10 --num_iters 100 --eval_dir experiments/eval/sage2/ --full_pass 1 --num_threads 20 --experiment_name sage2_bfs --pop_size 1 --very_small_test --validate_model None

# Random
python synthesis/learning.py experiments/configs/sage2/config_random.json --smt_batch_size 100 --benchmark_dir experiments/data/sage2/ --max_timeout 10 --num_iters 100 --eval_dir experiments/eval/sage2/ --full_pass 1 --num_threads 50 --experiment_name sage2_random --pop_size 1 --very_small_test --validate_model None

# FastText
python synthesis/learning.py experiments/configs/sage2/config_fast_text.json --benchmark_dir experiments/data/sage2/ --max_timeout 10 --num_iters 100 --eval_dir experiments/eval/sage2/ --smt_batch_size 100 --full_pass 1 --num_threads 40 --experiment_name sage2_bilinear --pop_size 1 --validate_model model_8.pt --very_small_test
mv experiments/eval/sage2/sage2_bilinear/valid/1 experiments/eval/sage2/sage2_bilinear/valid/10

# Evolutionary
python synthesis/learning.py experiments/configs/sage2/config_evo.json --smt_batch_size 100 --benchmark_dir experiments/data/sage2/ --max_timeout 10 --num_iters 100 --eval_dir experiments/eval/sage2/ --full_pass 1 --num_threads 20 --experiment_name sage2_evo --pop_size 1 --very_small_test --validate_model None

# NN
python synthesis/learning.py experiments/configs/sage2/config_apprentice.json --smt_batch_size 100 --benchmark_dir experiments/data/sage2/ --max_timeout 10 --num_iters 100 --eval_dir experiments/eval/sage2/ --full_pass 1 --num_threads 30 --experiment_name sage2_apprentice --pop_size 1 --very_small_test --validate_model model_10.pt
mv experiments/eval/sage2/sage2_apprentice/valid/1 experiments/eval/sage2/sage2_apprentice/valid/10
python synthesis/learning.py experiments/configs/sage2/config_apprentice.json --smt_batch_size 100 --benchmark_dir experiments/data/sage2/ --max_timeout 10 --num_iters 100 --eval_dir experiments/eval/sage2/ --full_pass 1 --num_threads 30 --experiment_name sage2_apprentice --pop_size 1 --very_small_test --validate_model model_8.pt
mv experiments/eval/sage2/sage2_apprentice/valid/1 experiments/eval/sage2/sage2_apprentice/valid/8
python synthesis/learning.py experiments/configs/sage2/config_apprentice.json --smt_batch_size 100 --benchmark_dir experiments/data/sage2/ --max_timeout 10 --num_iters 100 --eval_dir experiments/eval/sage2/ --full_pass 1 --num_threads 30 --experiment_name sage2_apprentice --pop_size 1 --very_small_test --validate_model model_6.pt
mv experiments/eval/sage2/sage2_apprentice/valid/1 experiments/eval/sage2/sage2_apprentice/valid/6
python synthesis/learning.py experiments/configs/sage2/config_apprentice.json --smt_batch_size 100 --benchmark_dir experiments/data/sage2/ --max_timeout 10 --num_iters 100 --eval_dir experiments/eval/sage2/ --full_pass 1 --num_threads 30 --experiment_name sage2_apprentice --pop_size 1 --very_small_test --validate_model model_4.pt
mv experiments/eval/sage2/sage2_apprentice/valid/1 experiments/eval/sage2/sage2_apprentice/valid/4
python synthesis/learning.py experiments/configs/sage2/config_apprentice.json --smt_batch_size 100 --benchmark_dir experiments/data/sage2/ --max_timeout 10 --num_iters 100 --eval_dir experiments/eval/sage2/ --full_pass 1 --num_threads 30 --experiment_name sage2_apprentice --pop_size 1 --very_small_test --validate_model model_2.pt
mv experiments/eval/sage2/sage2_apprentice/valid/1 experiments/eval/sage2/sage2_apprentice/valid/2

