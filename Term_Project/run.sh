python preprocessing.py -m lcg 10000 100
python train.py -m lcg 10000 50
python test.py -m lcg 10000 50 > result_lcg.txt

# python preprocessing.py -m random 200 100
# python train.py -m random 200 50
# python test.py -m random 200 50 > result_random.txt