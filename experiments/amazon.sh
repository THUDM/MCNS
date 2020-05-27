export CUDA_VISIBLE_DEVICES=0
python main.py --input ./data/amazon/ --epochs 50 --patience 20 --user_num 192403 --item_num 63001 --save_dir ./embeddings_amazon/
