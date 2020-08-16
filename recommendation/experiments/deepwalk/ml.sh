export CUDA_VISIBLE_DEVICES=2
python main.py --input ./data/ml-100k/ --model deepwalk --epochs 35 --batch_size 64 --validate_batch_size 64 --dim 256 --user_num 943 --item_num 1682 --patience 30 --save_dir ./embeddings_ml/
