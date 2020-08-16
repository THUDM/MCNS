export CUDA_VISIBLE_DEVICES=2
python main.py --input ./data/ali/ --model graphsage_mean --epochs 30 --patience 20 --user_num 106042 --item_num 53591 --save_dir ./embeddings_ali/