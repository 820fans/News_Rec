import numpy as np
from models.basemodel import basemodel
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from scipy import sparse
import matplotlib.pyplot as plt

class SVD_model(basemodel):
    '''SVD矩阵分解，设置隐式空间维度'''
    def __init__(self, user_news_df, dimk):
        basemodel.__init__(self, user_news_df)
        self.dimk = dimk

    # 将中间的矩阵转化为对角阵
    def vector_to_diagonal(self, vector):
        """
        将向量放在对角矩阵的对角线上
        :param vector:
        :return:
        """
        if (isinstance(vector, np.ndarray) and vector.ndim == 1) or \
                isinstance(vector, list):
            length = len(vector)
            diag_matrix = np.zeros((length, length))
            np.fill_diagonal(diag_matrix, vector)
            return diag_matrix
        return None

    def train(self):
        U, S, VT = svds(sparse.csr_matrix(self.ui_mat),  k=self.dimk, maxiter=1000) # 5个隐主题
        S = self.vector_to_diagonal(S)
        self.ui_pred = np.dot(np.dot(U, S), VT) * (self.ui_mat < 1e-6)

    def predict(self, user, item):
        """"""
        return self.ui_pred[user, item]


if __name__ == "__main__":
    # k= 5    -->   mPrecision: 0.00340974  mAP: 0.0092391   nDCG: 0.00496665
    # k= 6    -->   mPrecision: 0.00277937  mAP: 0.00776245  nDCG: 0.0042192
    # k=20    -->   mPrecision: 0.00260745  mAP: 0.00567252  nDCG: 0.00478021
    pass