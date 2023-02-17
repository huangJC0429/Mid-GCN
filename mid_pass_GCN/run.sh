python -u main_meta.py --dataset cora --alpha 0.5 --hid 256 --dropout 0.6 --ptb_rate 0.0
python -u main_meta.py --dataset citeseer --alpha 0.55 --hid 64 --dropout 0.6 --ptb_rate 0.0
python -u main_meta.py --dataset git --lr 0.05 -- epochs 300 --hid 64 --alpha 0.55 --ptb_rate 0.0
python -u main_meta.py --dataset polblogs --alpha 0.3 --lr 0.05 -- epochs 300 --hid 64 --alpha 0.5 --ptb_rate 0.0