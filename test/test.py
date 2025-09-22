import pandas as pd
import numpy as np
from utils.evaluate import accuracy
from config import BARCLAYS_CSV_24_25, BARCLAYS_CSV_21_24

train = pd.read_csv(BARCLAYS_CSV_21_24)
test = pd.read_csv(BARCLAYS_CSV_24_25)

train.to_json('train.json')
test.to_json('test.json')
