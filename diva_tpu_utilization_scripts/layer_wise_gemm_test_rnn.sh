model_list=( "rnn_lstm-small_seq32_mm.csv"
             "rnn_lstm-small_seq64_mm.csv"
            #  "rnn_lstm-small_seq128_mm.csv"
            #  "rnn_lstm-small_seq256_mm.csv"
            #  "rnn_lstm-large_seq32_mm.csv"
            #  "rnn_lstm-large_seq64_mm.csv"
            #  "rnn_lstm-large_seq128_mm.csv"
            #  "rnn_lstm-large_seq256_mm.csv"     
            )


for model in ${model_list[@]}; do
    python3 scripts/layer_wise_gemm_test_rnn.py layer_configs/$model
done