model_list=(# "transformer_bert-base_seq32_mm.csv"
             "transformer_bert-base_seq64_mm.csv"
            #  "transformer_bert-base_seq128_mm.csv"
            #  "transformer_bert-base_seq256_mm.csv"
            # "transformer_bert-large_seq32_mm.csv"
             "transformer_bert-large_seq64_mm.csv"
            #  "transformer_bert-large_seq128_mm.csv"
            #  "transformer_bert-large_seq256_mm.csv"     
            )


for model in ${model_list[@]}; do
    python3 scripts/layer_wise_gemm_test_transformer.py layer_configs/$model
done