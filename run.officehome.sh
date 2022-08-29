gpu=0

python main.py --cfg configs/officehome.yaml --gpu $gpu --log log/oh/LADA  ADA.AL LADA  ADA.DA ft
python main.py --cfg configs/officehome.yaml --gpu $gpu --log log/oh/LADA  ADA.AL LADA  ADA.DA mme
python main.py --cfg configs/officehome.yaml --gpu $gpu --log log/oh/LADA  ADA.AL LADA  ADA.DA LADA
python main.py --cfg configs/officehome.yaml --gpu $gpu --log log/oh/LADA  ADA.AL LADA  ADA.DA LADA  LADA.A_RAND_NUM 1
