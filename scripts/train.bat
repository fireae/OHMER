python train.py -model_type matrix -data data/crohme/demo -save_model demo-model -gpu_ranks 0 -batch_size 20 ^
                -max_grad_norm 20 -learning_rate 0.1 -encoder_type brnn -enc_input_size 6