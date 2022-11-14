python3 tutorials_new/per_layer_perf.py --dpsgd=False --vectorized --epochs 1 --batch_size 1024 --seq_length 16 --model lstm-small --n_batches 5
python3 tutorials_new/per_layer_perf.py --dpsgd=True --vectorized --epochs 1 
--batch_size 1024 --seq_length 16 --model lstm-small --n_batches 5
# python3 tutorials_new/per_layer_reweight_optimized.py --dpsgd=True --vectorized --epochs 1 --batch_size 1024 --seq_length 16 --model lstm-small --n_batches 5

python3 tutorials_new/per_layer_perf.py --dpsgd=False --vectorized --epochs 1 --batch_size 2048 --seq_length 16 --model lstm-small --n_batches 5
python3 tutorials_new/per_layer_perf.py --dpsgd=True --vectorized --epochs 1 --batch_size 2048 --seq_length 16 --model lstm-small --n_batches 5
# python3 tutorials_new/per_layer_reweight_optimized.py --dpsgd=True --vectorized --epochs 1 --batch_size 2048 --seq_length 16 --model lstm-small --n_batches 5

python3 tutorials_new/per_layer_perf.py --dpsgd=False --vectorized --epochs 1 --batch_size 4096 --seq_length 16 --model lstm-small --n_batches 5
python3 tutorials_new/per_layer_perf.py --dpsgd=True --vectorized --epochs 1 --batch_size 4096 --seq_length 16 --model lstm-small --n_batches 5
# python3 tutorials_new/per_layer_reweight_optimized.py --dpsgd=True --vectorized --epochs 1 --batch_size 4096 --seq_length 16 --model lstm-small --n_batches 5