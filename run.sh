
# cora
python citation.py --dataset cora --degree 10 --weight_decay 2e-5 --T 10 --alpha 0.9
# citeseer
python citation.py --dataset citeseer --epochs 150 --degree 8 --weight_decay 2e-4 --T 5 --alpha 1.1
# pubmed
python citation.py --dataset pubmed --degree 14 --weight_decay 1e-5 --T 20 --alpha 2.2


# faster version
# cora
python citation.py --dataset cora --degree 10 --weight_decay 2e-5 --T 10 --alpha 0.9 --fast
# citeseer
python citation.py --dataset citeseer --epochs 150 --degree 8 --weight_decay 2e-4 --T 5 --alpha 1.1 --fast
# pubmed
python citation.py --dataset pubmed --degree 14 --weight_decay 1e-5 --T 20 --alpha 2.2 --fast