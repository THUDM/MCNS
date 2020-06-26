export CUDA_VISIBLE_DEVICES=1
python main.py --input ./data/amazon/ --model graphsage_mean --epochs 30 --patience 20 --user_num 192403 --item_num 63001 --save_dir ./embeddings_amazon/
