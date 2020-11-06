import re
import time
import numpy as np
import urllib.parse
import math
from sklearn import model_selection
from sklearn import svm
import joblib
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


# 载入数据和标签
def load_data(file_name, label):
    datas = []
    with open(file_name, 'r', encoding="utf-8") as f:
        for line in f:
            datas.append(str(urllib.parse.unquote(line)))
    labels = np.full(len(datas), label)
    return datas, labels


def extf(data, tmpf, pattern_list, weigh):
    for x in pattern_list:
        # print(x);
        aaa = len(re.findall(x, data))
        lll = np.log(1.0 + aaa)
        tmpf.append(lll * weigh)


def iextf(data, tmpf, pattern_list, weigh):
    for x in pattern_list:
        # print(x);
        aaa = len(re.findall(x, data, re.IGNORECASE))
        lll = np.log(1.0 + aaa)
        tmpf.append(lll * weigh)

def get_feature(datas):
    f0 = []
    for data in datas:
        tmpf = []
        tmpf.append(len(data))
        if re.search('(http://)|(https://)', data, re.IGNORECASE):
            tmpf.append(1)
        else:
            tmpf.append(0)
        extf(data, tmpf, ["test"], 0.0001)

        extf(data, tmpf, ["<", ">", "[<>]", "\"", "\'", "[{}]", "{", "}", "\(\)", "\(", "\)"], 1)
        tmpf.append(int(len(re.findall("[<]", data)) == len(re.findall("[>]", data))))
        tmpf.append(int(len(re.findall("[\']", data)) % 2 == 0))
        tmpf.append(int(len(re.findall("[\"]", data)) % 2 == 0))
        tmpf.append(int(len(re.findall("[{]", data)) == len(re.findall("[|]", data))))
        tmpf.append(int(len(re.findall("\(", data)) == len(re.findall("\)", data))))
        extf(data, tmpf, ["\$", "\.", "=", "|", "&", ";", "\?", "%", "#", "\[", "\]", "/"], 1)
        iextf(data, tmpf, ["\$", "\.", "=", "|", "&", ";", "\?", "%", "#", "\[", "\]", "/"], 1)

        extf(data, tmpf, ["..", "../"], 1)
        iextf(data, tmpf, ["..", "../"], 1)

        extf(data, tmpf, ["document", "eval", "phpinfo"], 1)
        iextf(data, tmpf, ["document", "eval", "phpinfo"], 1)

        extf(data, tmpf, ["script"], 2)
        iextf(data, tmpf, ["script"], 2)

        extf(data, tmpf, ["<script>"], 3)
        iextf(data, tmpf, ["<script>"], 3)
        extf(data, tmpf, ["</script>"], 3)
        iextf(data, tmpf, ["</script>"], 3)

        extf(data, tmpf, ["getElement", "alert", "javascript", "onerror", "onload"], 3)
        iextf(data, tmpf, ["getElement", "alert", "javascript", "onerror", "onload"], 3)

        extf(data, tmpf, ["on", "src", "src=", "exit"], 1)
        iextf(data, tmpf, ["on", "src", "src=", "exit"], 1)

        extf(data, tmpf, ["print", "assert", "preg_replace", "cookie", "exe", "/etc", "sql", "admin", "manage", "root"],
             1)
        iextf(data, tmpf,
              ["print", "assert", "preg_replace", "cookie", "exe", "/etc", "sql", "admin", "manage", "root"], 1)

        extf(data, tmpf, ["pl", "dll", "jsp", "asp", "php"], 1)
        iextf(data, tmpf, ["pl", "dll", "jsp", "asp", "php"], 1)

        extf(data, tmpf,
             ["/\.\./", "/\./", "select", "create", "update", "set-cookie", "password", "passwd", "pass", "<\[a-zA-Z]"],
             1)
        iextf(data, tmpf, ["/\.\./", "/\./", "select", "create", "update", "set-cookie", "password", "passwd", "pass",
                           "<\[a-zA-Z]"], 1)

        f0.append(tmpf)
    # feature = np.reshape(url_len + url_count + evil_char+ evil_char*evil_char + evil_word +evil_word*evil_word, (4, len(url_len))).transpose()#四个特征
    feature = np.array(f0)
    return feature


def show_predict(test_datas, test_labels, modelfile_name):
    model = joblib.load(modelfile_name)
    data_num = len(test_datas)  # 得到样本数
    index = np.arange(data_num)  # 生成下标
    np.random.shuffle(index)

    test_datas = np.array(test_datas)[index]
    test_labels = np.array(test_labels)[index]

    x_data = get_feature(test_datas[:400])

    test_labels = test_labels[:400]

    result = model.predict(x_data)
    # for i in range(0, len(result)):
    #     if result[i] != test_labels[i]:
    #         print(test_datas[i])
    show_result(test_labels, result)


def show_result(y_test, y_pred):
    print("测试数据准确率:")
    print(metrics.accuracy_score(y_test, y_pred))
    print("结果矩阵:")
    print(metrics.confusion_matrix(y_test, y_pred))
    print("正确率:")
    print(metrics.precision_score(y_test, y_pred))
    print("召回率:")
    print(metrics.recall_score(y_test, y_pred))
    print("F1值:")
    print(metrics.f1_score(y_test, y_pred))


def LR_model(datas, labels):
    model_name = "model_lr.pkl"
    x_train, x_test, y_train, y_test = model_selection.train_test_split(datas, labels, test_size=0.4, random_state=0)
    lgs = LogisticRegression(class_weight={1: 2 * 40000 / 100000, 0: 1.0}, C=1.2, penalty='l2').fit(x_train, y_train)
    joblib.dump(lgs, model_name)
    result = lgs.predict(x_test)
    show_result(y_test, result)


if __name__ == '__main__':
    bad_dataset, bad_labels = load_data("data/test_badqueries.txt", 0)
    good_dataset, good_labels = load_data("data/test_goodqueries.txt", 1)
    # datas = bad_dataset[:40000] + good_dataset[:100000]
    # labels = np.concatenate((bad_labels[:40000], good_labels[:100000]), axis=0)
    # feature = get_feature(datas)
    print(bad_labels,good_labels)
    model = joblib.load("model_lr.pkl")
    while(1):
        url= input()
        data = []
        data.append(url)
        x_data = get_feature(data)

        result = model.predict(x_data)
        print(result)
    # test_datas = bad_dataset[:20] + good_dataset[:20]
    # test_labels = np.concatenate((bad_labels[:20],good_labels[:20]), axis=0)
    # show_predict(test_datas, test_labels, "model_lr.pkl")
    print("feature exact done")
   # print(feature[:10])
    # LR_model(feature, labels)
    # import code
    #
    # interp = code.InteractiveConsole(globals())
    # interp.interact("")