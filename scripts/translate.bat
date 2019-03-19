python translate.py -data_type matrix -model demo-model_step_25000.pt -src_dir data/crohme/data ^
                    -src data/crohme/src-test.txt -tgt data/crohme/tgt-test.txt -output pred.txt ^
                    -shard_size 100 -max_length 150 -beam_size 5 -gpu 0 ^
                    -batch_size 10 -verbose -report_bleu