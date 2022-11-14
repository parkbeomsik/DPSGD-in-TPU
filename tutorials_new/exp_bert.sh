python3 tutorials_new/dpsgd_transformer_perf.py --dpsgd=True --vectorized --epochs 1 --batch_size 4 --seq_length 32 --model bert-base --n_batches 5
python3 tutorials_new/dpsgd_transformer_perf.py --dpsgd=False --vectorized --epochs 1 --batch_size 4 --seq_length 32 --model bert-base --n_batches 5
python3 tutorials_new/dpsgd_reweight_transformer.py --dpsgd=True --vectorized --epochs 1 --batch_size 4 --seq_length 32 --model bert-base --n_batches 5

python3 tutorials_new/dpsgd_transformer_perf.py --dpsgd=True --vectorized --epochs 1 --batch_size 2 --seq_length 32 --model bert-large --n_batches 5
python3 tutorials_new/dpsgd_transformer_perf.py --dpsgd=False --vectorized --epochs 1 --batch_size 2 --seq_length 32 --model bert-large --n_batches 5
python3 tutorials_new/dpsgd_reweight_transformer.py --dpsgd=True --vectorized --epochs 1 --batch_size 2 --seq_length 32 --model bert-large --n_batches 5

python3 tutorials_new/dpsgd_reweight_transformer.py --dpsgd=True --vectorized --epochs 1 --batch_size 4 --seq_length 32 --model bert-large --n_batches 5