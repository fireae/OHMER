python train.py -model_type matrix -decoder_type transformer -data data/crohme/demo ^
                -save_model demo-model -log_file train.log -tensorboard ^
                -learning_rate 1.0 -batch_size 20 -max_grad_norm 20 -param_init_glorot^
                -enc_input_size 9 -enc_rnn_size 480 -dec_rnn_size 480 -tgt_word_vec_size 480 ^
                -save_checkpoint_steps 1000 -valid_steps 2000 -train_steps 20000 ^
                -gpu_ranks 0