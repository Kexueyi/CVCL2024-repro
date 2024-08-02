# python main.py --batch_size 512 --device cuda:1 &
# python main.py --model clip --batch_size 512 --device cuda:0
python main.py --model cvcl_res --use_attr True
python main.py --model cvcl_vit --use_attr True
python main.py --model clip --use_attr True

python main.py --model cvcl_res --use_attr True --prefix "a photo of a "
python main.py --model cvcl_vit --use_attr True --prefix "a photo of a "
python main.py --model clip --use_attr True --prefix "a photo of a "


