python zs_base.py --batch_size 512 --device cuda:1 &
python zs_base.py --model clip --batch_size 256 --device cuda:0

sleep 10

python zs_base.py --model cvcl_vit --batch_size 512 --device cuda:0
