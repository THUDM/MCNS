export CUDA_VISIBLE_DEVICES=3
python main.py --input ./data/ml-100k/ --epochs 200 --batch_size 256 --validate_batch_size 256 --samples_1 50 --samples_2 20 --dim_1 128 --dim_2 128 --max_degree 500 --user_num 943 --item_num 1682 --patience 100 --save_dir ./embeddings_ml/
