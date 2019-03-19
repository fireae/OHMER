python preprocess.py -data_type matrix -src_dir data/crohme/data/ -train_src data/crohme/src-train.txt ^
                     -train_tgt data/crohme/tgt-train.txt -valid_src data/crohme/src-val.txt ^
                     -valid_tgt data/crohme/tgt-val.txt -save_data data/crohme/demo ^
                     -tgt_vocab data/crohme/vocab.txt -shard_size 1000