import urllib.parse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import torch
import joblib
from numpy import array

from numpy import argmax
import torch.nn.functional
def get_ngrams(query):
    tempQuery = str(query)
    ngrams = []
    for i in range(0, len(tempQuery) - 3):
        ngrams.append(tempQuery[i:i + 3])
    return ngrams

def get_vectorizer():
    vectorizer = TfidfVectorizer(tokenizer=get_ngrams)
    return vectorizer

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

def get_feature1(data,vectorizer):
    datas = vectorizer.fit_transform(data)
    return datas

def get_datas(vectorizer):
    bdata,blabel= get_data("data/badqueries.txt",0,23333)
    gdata,glabel = get_data("data/goodqueries.txt",1,88888)
    # datas = bdata+gdata
    labels = blabel+glabel
    data = bdata+gdata

    datas = get_feature1(data,vectorizer)
    print(datas.shape)

    train_data, test_data,train_label, test_label= train_test_split(datas, labels, test_size=20, random_state=42)
    return train_data, test_data,train_label, test_label

def get_batch(dataset,batch_n):
    arr = np.random.randint(0,len(dataset),size=(batch_n,))
    batch = [dataset[i] for i in arr]
    print(arr)
    print(batch)

def ont_hot(batch):

    alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ`~!@#$%^&*()_+=-{}|\][:\"\';?><,./ 1234567890'
    # define a mapping of chars to integers
    print(len(alphabet))
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    print(char_to_int)
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    # integer encode input data
    batch_encoded = []
    for data in batch:
        integer_encoded = [char_to_int[char] for char in data]
        batch_encoded.append(integer_encoded)
    print(batch)
    # one hot encode

    batch_one_hot = []
    for integer_encoded in batch_encoded:
        temp = np.zeros((200, 95))
        for i in range(200):
            # letter = [0 for _ in range(len(alphabet))]
            if i<len(integer_encoded):
               temp[i][integer_encoded[i]] = 1
            # onehot_encoded.append(letter)
        batch_one_hot.append(temp)
    one_hot = torch.Tensor(batch_one_hot)
    print(one_hot,one_hot.shape)
    # invert encoding
    # inverted = int_to_char[argmax(onehot_encoded[0])]

    return one_hot
    # return one_hot


def load_LGmodel(model_name,vectorizer_name):
    model = joblib.load(model_name)
    vectorizer = joblib.load(vectorizer_name)
    return model,vectorizer

def load_model(path):
    pass




if __name__=="__main__":
    batch = [ 'dassdqwdwqd/312311980)@Q*#)!*$:','dassdqwdwqd/312311980)@Q*#)!*$:dsada']
    ont_hot(batch)
    # dataset = np.arange(0,100,dtype=np.float32)
    # get_batch(dataset,32)

