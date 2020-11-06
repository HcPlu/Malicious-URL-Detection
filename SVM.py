import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from common import get_datas,get_vectorizer

# def load_data(filename):
#     '''
#     假设这是鸢尾花数据,csv数据格式为：
#     0,5.1,3.5,1.4,0.2
#     0,5.5,3.6,1.3,0.5
#     1,2.5,3.4,1.0,0.5
#     1,2.8,3.2,1.1,0.2
#     每一行数据第一个数字(0,1...)是标签,也即数据的类别。
#     '''
#     data = np.genfromtxt(filename, delimiter=',')
#     x = data[:, 1:]  # 数据特征
#     y = data[:, 0].astype(int)  # 标签
#     scaler = StandardScaler()
#     x_std = scaler.fit_transform(x)  # 标准化
#     # 将数据划分为训练集和测试集，test_size=.3表示30%的测试集
#     x_train, x_test, y_train, y_test = train_test_split(x_std, y, test_size=.3)
#     return x_train, x_test, y_train, y_test


def svm_c(x_train, x_test, y_train, y_test):
    # rbf核函数，设置数据权重
    svc = SVC(kernel='rbf', class_weight='balanced',)
    c_range = np.logspace(-5, 15, 11, base=2)
    gamma_range = np.logspace(-9, 3, 13, base=2)
    # 网格搜索交叉验证的参数范围，cv=3,3折交叉
    param_grid = [{'kernel': ['rbf'], 'C': c_range, 'gamma': gamma_range}]
    grid = GridSearchCV(svc, param_grid, cv=3, n_jobs=-1)
    # 训练模型
    clf = grid.fit(x_train, y_train)
    # 计算测试集精度
    score = grid.score(x_test, y_test)
    print('精度为%s' % score)

if __name__ == '__main__':
    svm_c(*get_datas(get_vectorizer()))