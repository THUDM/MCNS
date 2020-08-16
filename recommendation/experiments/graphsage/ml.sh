export CUDA_VISIBLE_DEVICES=2
python main.py --input ./data/ml-100k/ --model graphsage_mean --epochs 200 --batch_size 256 --validate_batch_size 256 --dim_1 128 --dim_2 128 --user_num 943 --item_num 1682 --patience 100 --save_dir ./embeddings_ml/
