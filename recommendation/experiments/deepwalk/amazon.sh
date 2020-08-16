export CUDA_VISIBLE_DEVICES=6
python main.py --input ./data/amazon/ --model deepwalk --epochs 30 --batch_size 64 --validate_batch_size 64 --dim 512 --patience 20 --user_num 192403 --item_num 63001 --save_dir ./embeddings_amazon/
