python3 tutorials_new/dpsgd_cnn_perf.py --dpsgd=True --vectorized --epochs 1 --batch_size 1328 --input_size 32 --model squeezenetv1.1 --n_batches 5
python3 tutorials_new/dpsgd_cnn_perf.py --dpsgd=False --vectorized --epochs 1 --batch_size 1328 --input_size 32 --model squeezenetv1.1 --n_batches 5
python3 tutorials_new/dpsgd_reweight_cnn.py --dpsgd=True --vectorized --epochs 1 --batch_size 1328 --input_size 32 --model squeezenetv1.1 --n_batches 5

python3 tutorials_new/dpsgd_cnn_perf.py --dpsgd=True --vectorized --epochs 1 --batch_size 2656 --input_size 32 --model squeezenetv1.1 --n_batches 5
python3 tutorials_new/dpsgd_cnn_perf.py --dpsgd=False --vectorized --epochs 1 --batch_size 2656 --input_size 32 --model squeezenetv1.1 --n_batches 5
python3 tutorials_new/dpsgd_reweight_cnn.py --dpsgd=True --vectorized --epochs 1 --batch_size 2656 --input_size 32 --model squeezenetv1.1 --n_batches 5

python3 tutorials_new/dpsgd_cnn_perf.py --dpsgd=True --vectorized --epochs 1 --batch_size 640 --input_size 32 --model mobilenetv3small --n_batches 5
python3 tutorials_new/dpsgd_cnn_perf.py --dpsgd=False --vectorized --epochs 1 --batch_size 640 --input_size 32 --model mobilenetv3small --n_batches 5
python3 tutorials_new/dpsgd_reweight_cnn.py --dpsgd=True --vectorized --epochs 1 --batch_size 640 --input_size 32 --model mobilenetv3small --n_batches 5

python3 tutorials_new/dpsgd_cnn_perf.py --dpsgd=True --vectorized --epochs 1 --batch_size 1280 --input_size 32 --model mobilenetv3small --n_batches 5
python3 tutorials_new/dpsgd_cnn_perf.py --dpsgd=False --vectorized --epochs 1 --batch_size 1280 --input_size 32 --model mobilenetv3small --n_batches 5
python3 tutorials_new/dpsgd_reweight_cnn.py --dpsgd=True --vectorized --epochs 1 --batch_size 1280 --input_size 32 --model mobilenetv3small --n_batches 5