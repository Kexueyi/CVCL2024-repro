# Baseline, no attribute, no baby_vocab
python main.py --model cvcl_res --use_attr False --baby_vocab False
python main.py --model cvcl_vit --use_attr False --baby_vocab False
python main.py --model clip --use_attr False --baby_vocab False
# Filter baby, no attribute
python main.py --model cvcl_res --use_attr False 
python main.py --model cvcl_vit --use_attr False 
python main.py --model clip --use_attr False 
# with attribute, top_3
python main.py --model cvcl_res  --top_n_desc 3
python main.py --model cvcl_vit --top_n_desc 3
python main.py --model clip --top_n_desc 3
# with attribute, top_5
python main.py --model cvcl_res  
python main.py --model cvcl_vit
python main.py --model clip
# with attribute, top_8
python main.py --model cvcl_res  --top_n_desc 8
python main.py --model cvcl_vit --top_n_desc 8
python main.py --model clip --top_n_desc 8
# with attribute, top_11
python main.py --model cvcl_res  --top_n_desc 11
python main.py --model cvcl_vit --top_n_desc 11
python main.py --model clip --top_n_desc 11



