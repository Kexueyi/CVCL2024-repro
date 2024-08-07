# Baseline, no attribute, no baby_vocab
python main.py --model cvcl_res 
python main.py --model cvcl_vit 
python main.py --model clip 
# with attribute, top_3
python main.py --model cvcl_res  --use_attr --top_n_desc 3
python main.py --model cvcl_vit --use_attr --top_n_desc 3
python main.py --model clip --use_attr --top_n_desc 3
# with attribute, top_5
python main.py --model cvcl_res  --use_attr --top_n_desc 5
python main.py --model cvcl_vit --use_attr --top_n_desc 5
python main.py --model clip --use_attr --top_n_desc 5
# # with attribute, top_8
# python main.py --model cvcl_res  --use_attr --top_n_desc 8
# python main.py --model cvcl_vit --use_attr --top_n_desc 8
# python main.py --model clip --use_attr --top_n_desc 8
# with attribute, top_11
python main.py --model cvcl_res --use_attr --top_n_desc 11
python main.py --model cvcl_vit --use_attr --top_n_desc 11
python main.py --model clip --use_attr --top_n_desc 11

# Filter baby, no attribute
python main.py --model cvcl_res --baby_vocab  
python main.py --model cvcl_vit --baby_vocab  
python main.py --model clip --baby_vocab  

python main.py --model cvcl_res --baby_vocab  --use_attr --top_n_desc 3
python main.py --model cvcl_vit --baby_vocab  --use_attr --top_n_desc 3
python main.py --model clip --baby_vocab  --use_attr --top_n_desc 3

python main.py --model cvcl_res --baby_vocab  --use_attr --top_n_desc 5
python main.py --model cvcl_vit --baby_vocab  --use_attr --top_n_desc 5
python main.py --model clip --baby_vocab  --use_attr --top_n_desc 5

python main.py --model cvcl_res --baby_vocab  --use_attr --top_n_desc 11
python main.py --model cvcl_vit --baby_vocab  --use_attr --top_n_desc 11
python main.py --model clip --baby_vocab  --use_attr --top_n_desc 11


