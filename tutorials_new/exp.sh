# python3 tutorials_new/dpsgd_cnn_perf.py --dpsgd=True --vectorized --epochs 1 --batch_size 56 --input_size 32 --model vgg16 --n_batches 5
# python3 tutorials_new/dpsgd_cnn_perf.py --dpsgd=False --vectorized --epochs 1 --batch_size 56 --input_size 32 --model vgg16 --n_batches 5
# python3 tutorials_new/dpsgd_reweight_cnn.py --dpsgd=True --vectorized --epochs 1 --batch_size 56 --input_size 32 --model vgg16 --n_batches 5
# python3 tutorials_new/dpsgd_reweight_cnn_optimized.py --dpsgd=True --vectorized --epochs 1 --batch_size 56 --input_size 32 --model vgg16 --n_batches 5

# python3 tutorials_new/dpsgd_cnn_perf.py --dpsgd=True --vectorized --epochs 1 --batch_size 84 --input_size 32 --model resnet50 --n_batches 5
# python3 tutorials_new/dpsgd_cnn_perf.py --dpsgd=False --vectorized --epochs 1 --batch_size 84 --input_size 32 --model resnet50 --n_batches 5
# python3 tutorials_new/dpsgd_reweight_cnn.py --dpsgd=True --vectorized --epochs 1 --batch_size 84 --input_size 32 --model resnet50 --n_batches 5

# python3 tutorials_new/dpsgd_cnn_perf.py --dpsgd=True --vectorized --epochs 1 --batch_size 33 --input_size 32 --model resnet152 --n_batches 5
# python3 tutorials_new/dpsgd_cnn_perf.py --dpsgd=False --vectorized --epochs 1 --batch_size 33 --input_size 32 --model resnet152 --n_batches 5
# python3 tutorials_new/dpsgd_reweight_cnn.py --dpsgd=True --vectorized --epochs 1 --batch_size 33 --input_size 32 --model resnet152 --n_batches 5

# python3 tutorials_new/dpsgd_cnn_perf.py --dpsgd=True --vectorized --epochs 1 --batch_size 2656 --input_size 32 --model squeezenetv1.1 --n_batches 5
# python3 tutorials_new/dpsgd_cnn_perf.py --dpsgd=False --vectorized --epochs 1 --batch_size 2656 --input_size 32 --model squeezenetv1.1 --n_batches 5
# python3 tutorials_new/dpsgd_reweight_cnn.py --dpsgd=True --vectorized --epochs 1 --batch_size 2656 --input_size 32 --model squeezenetv1.1 --n_batches 5

# python3 tutorials_new/dpsgd_cnn_perf.py --dpsgd=True --vectorized --epochs 1 --batch_size 1280 --input_size 32 --model mobilenetv3small --n_batches 5
# python3 tutorials_new/dpsgd_cnn_perf.py --dpsgd=False --vectorized --epochs 1 --batch_size 1280 --input_size 32 --model mobilenetv3small --n_batches 5
# python3 tutorials_new/dpsgd_reweight_cnn.py --dpsgd=True --vectorized --epochs 1 --batch_size 1280 --input_size 32 --model mobilenetv3small --n_batches 5

# python3 tutorials_new/dpsgd_transformer_perf.py --dpsgd=True --vectorized --epochs 1 --batch_size 8 --seq_length 32 --model bert-base --n_batches 5
# python3 tutorials_new/dpsgd_transformer_perf.py --dpsgd=False --vectorized --epochs 1 --batch_size 8 --seq_length 32 --model bert-base --n_batches 5
# python3 tutorials_new/dpsgd_reweight_transformer.py --dpsgd=True --vectorized --epochs 1 --batch_size 8 --seq_length 32 --model bert-base --n_batches 5

# python3 tutorials_new/dpsgd_transformer_perf.py --dpsgd=True --vectorized --epochs 1 --batch_size 4 --seq_length 32 --model bert-large --n_batches 5
# python3 tutorials_new/dpsgd_transformer_perf.py --dpsgd=False --vectorized --epochs 1 --batch_size 4 --seq_length 32 --model bert-large --n_batches 5
# python3 tutorials_new/dpsgd_reweight_transformer.py --dpsgd=True --vectorized --epochs 1 --batch_size 4 --seq_length 32 --model bert-large --n_batches 5

# python3 tutorials_new/dpsgd_transformer_perf.py --dpsgd=True --vectorized --epochs 1 --batch_size 8 --seq_length 32 --model bert-base --n_batches 5
# python3 tutorials_new/dpsgd_transformer_perf.py --dpsgd=False --vectorized --epochs 1 --batch_size 8 --seq_length 32 --model bert-base --n_batches 5
# python3 tutorials_new/dpsgd_reweight_transformer.py --dpsgd=True --vectorized --epochs 1 --batch_size 8 --seq_length 32 --model bert-base --n_batches 5

# python3 tutorials_new/dpsgd_transformer_perf.py --dpsgd=True --vectorized --epochs 1 --batch_size 4 --seq_length 32 --model bert-large --n_batches 5
# python3 tutorials_new/dpsgd_transformer_perf.py --dpsgd=False --vectorized --epochs 1 --batch_size 4 --seq_length 32 --model bert-large --n_batches 5
# python3 tutorials_new/dpsgd_reweight_transformer.py --dpsgd=True --vectorized --epochs 1 --batch_size 4 --seq_length 32 --model bert-large --n_batches 5

python3 tutorials_new/dpsgd_lstm_perf.py --dpsgd=True --vectorized --epochs 1 --batch_size 360 --seq_length 32 --model lstm-small --n_batches 5
# python3 tutorials_new/dpsgd_lstm_perf.py --dpsgd=False --vectorized --epochs 1 --batch_size 360 --seq_length 32 --model lstm-small --n_batches 5
# python3 tutorials_new/dpsgd_reweight_lstm.py --dpsgd=True --vectorized --epochs 1 --batch_size 360 --seq_length 32 --model lstm-small --n_batches 5

python3 tutorials_new/dpsgd_lstm_perf.py --dpsgd=True --vectorized --epochs 1 --batch_size 19 --seq_length 32 --model lstm-large --n_batches 5
# python3 tutorials_new/dpsgd_lstm_perf.py --dpsgd=False --vectorized --epochs 1 --batch_size 19 --seq_length 32 --model lstm-large --n_batches 5
# python3 tutorials_new/dpsgd_reweight_lstm.py --dpsgd=True --vectorized --epochs 1 --batch_size 19 --seq_length 32 --model lstm-large --n_batches 5