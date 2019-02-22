import numpy as np
from models.basemodel import basemodel
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF

class NMF_model(basemodel):
    '''NMF非负矩阵分解，设置隐式空间维度'''
    def __init__(self, user_news_df, dimk):
        basemodel.__init__(self, user_news_df)
        self.dimk = dimk
        self.nmf = NMF(n_components=self.dimk)

    def train(self):
        self.user_factors = self.nmf.fit_transform(self.ui_mat)
        self.item_factors = self.nmf.components_
        # print '用户的主题分布：'
        # print user_distribution
        # print '物品的主题分布：'
        # print item_distribution

    def predict(self, user, item):
        """"""
        u_f = self.user_factors[user, :]
        i_f = self.item_factors[:, item]
        prediction = np.matmul(u_f, i_f)
        return prediction


if __name__ == "__main__":
    # k=5   -->  mPrecision: 0.00323782  mAP: 0.00889423  nDCG: 0.00502365
    # k=20  -->  mPrecision: 0.00226361  mAP: 0.00475576  nDCG: 0.00413429
    # k=50  -->  mPrecision: 0.00140401  mAP: 0.00206008  nDCG: 0.00258748
    pass