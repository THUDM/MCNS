export CUDA_VISIBLE_DEVICES=3
python main.py --input ./data/ml-100k/ --epochs 100 --batch_size 256 --validate_batch_size 256 --dim_1 128 --dim_2 128 --user_num 943 --item_num 1682 --patience 50 --save_dir ./embeddings_ml/
