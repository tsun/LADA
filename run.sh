
# office-home 5%-budget
python main.py --cfg configs/officehome.yaml --gpu 0 --log log/oh/LADA ADA.AL LAS ADA.DA ft
python main.py --cfg configs/officehome.yaml --gpu 0 --log log/oh/LADA ADA.AL LAS ADA.DA mme
python main.py --cfg configs/officehome.yaml --gpu 0 --log log/oh/LADA ADA.AL LAS ADA.DA RAA
python main.py --cfg configs/officehome.yaml --gpu 0 --log log/oh/LADA ADA.AL LAS ADA.DA LAA

# office-home 10%-budget
python main.py --cfg configs/officehome.yaml --gpu 0 --log log/oh/LADA ADA.AL LAS ADA.DA RAA LADA.S_M 5 ADA.BUDGET 0.1
python main.py --cfg configs/officehome.yaml --gpu 0 --log log/oh/LADA ADA.AL LAS ADA.DA LAA LADA.S_M 5 ADA.BUDGET 0.1

# office-home rsut 10%-budget
python main.py --cfg configs/officehome_RSUT.yaml --gpu 0 --log log/oh_RSUT/LADA ADA.AL LAS ADA.DA RAA LADA.S_M 5 ADA.BUDGET 0.1
python main.py --cfg configs/officehome_RSUT.yaml --gpu 0 --log log/oh_RSUT/LADA ADA.AL LAS ADA.DA LAA LADA.S_M 5 ADA.BUDGET 0.1

# office-31 5%-budget
python main.py --cfg configs/office31.yaml --gpu 0 --log log/office31/LADA ADA.AL LAS ADA.DA LAA

# visda 5%-budget
python main.py --cfg configs/visda.yaml --gpu 0 --log log/visda/LADA ADA.AL LAS ADA.DA LAA


