import sklearn
import numpy as np
import urllib.parse
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import time

def get_data(path, label, number):
    f = open(path,encoding="utf-8")
    # data = f.readline()
    datas = []
    for line in f:
        print(line)
        datas.append(str(urllib.parse.unquote(line)))
    labels = [label for _ in range(number)]
    f.close()
    return datas[:number], labels

def get_ngrams(query):
    tempQuery = str(query)
    ngrams = []
    for i in range(0, len(tempQuery) - 3):
        ngrams.append(tempQuery[i:i + 3])
    return ngrams


def get_feature1(data):

    datas = vectorizer.fit_transform(data)

    return datas


def train(model_name):
    bdata,blabel= get_data("data/badqueries.txt",0,23333)
    gdata,glabel = get_data("data/goodqueries.txt",1,88888)
    # datas = bdata+gdata
    labels = blabel+glabel
    data = bdata+gdata

    datas = get_feature1(data)
    print(datas.shape)

    train_data, test_data,train_label, test_label= train_test_split(datas, labels, test_size=5000, random_state=7)
    LR = LogisticRegression(class_weight={1: 2 * 40000 / 100000, 0: 1.0}, C=1.2, penalty='l2')

    # Train
    LR.fit(train_data, train_label)
    joblib.dump(LR, model_name)
    joblib.dump(vectorizer, 'vectorizer_' + str(Time) + '.pkl')
    print('Model accuracy:{}'.format(LR.score(test_data, test_label)))
    # print("\n")

def load_model(model_name,vectorizer_name):
    model = joblib.load(model_name)
    vectorizer = joblib.load(vectorizer_name)
    return model,vectorizer

def test(data,model_name,vectorizer_name):
    model,vectorizer= load_model(model_name,vectorizer_name)
    # test_data = get_feature2(data)
    test_data = vectorizer.transform(data)
    result = model.predict(test_data)
    return result



if __name__ == '__main__':
    Time = time.time()
    vectorizer = TfidfVectorizer(tokenizer=get_ngrams)
    # train('SVM'+str(Time)+'.pkl')
    model_name = 'SVM1604505740.8325007.pkl'
    vectorizer_name = 'vectorizer_1604505740.8325007.pkl'
    data = ['www.baidu.com']
    print(test(data,model_name,vectorizer_name))