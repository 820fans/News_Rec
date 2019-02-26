from models.basemodel import basemodel
from models.UserCF import userCF
from models.NMF_model import NMF_model
from models.SVD_model import SVD_model
import pandas as pd
import profile

import numpy as np

# print(np.__version__)

data_df = pd.read_csv("Data/train_data.txt", sep='\t', header=-1)
test_df = pd.read_csv("Data/test_data.txt", sep='\t', header=-1)
# model = NMF_model(data_df, 50)
# model = NMF_model(data_df, 20)
# model = userCF(data_df, 5)
model = SVD_model(data_df, 6)
model.train()

model.evaluation(test_df)

# best:  [0.39546783625731041, 0.37456140350877132, 0.31491228070175387, 0.41066404399874989, 0.39764448720509904,
#         0.37044137039933167]
# epoch
# 14
# gen:  [0.39546783625731041, 0.37456140350877132, 0.31491228070175387, 0.41014948399214601, 0.39727260101510603,
#        0.37019775501297353]
