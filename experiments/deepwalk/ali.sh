export CUDA_VISIBLE_DEVICES=6
python main.py --input ./data/ali/ --model deepwalk --epochs 15 --patience 10 --user_num 106042 --item_num 53591 --save_dir ./embeddings_ali/
