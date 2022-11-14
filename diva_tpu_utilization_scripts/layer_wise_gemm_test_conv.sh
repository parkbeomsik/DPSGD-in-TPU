model_list=("cnn_vgg16_mnist_mm.csv"
            #  "cnn_vgg16_imagenet_mm.csv"
            #  "cnn_resnet50_mnist_mm.csv"
            #  "cnn_resnet50_imagenet_mm.csv"
            #  "cnn_resnet152_mnist_mm.csv"
            #  "cnn_resnet152_imagenet_mm.csv"
            #  "cnn_squeezenet_mnist_mm.csv"
            #  "cnn_squeezenet_imagenet_mm.csv"
             "cnn_mobilenet_mnist_mm.csv"
            #  "cnn_mobilenet_imagenet_mm.csv"        
            )


for model in ${model_list[@]}; do
    python3 scripts/layer_wise_gemm_test_conv.py layer_configs/$model
done