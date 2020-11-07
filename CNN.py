
# https://keras-cn-docs.readthedocs.io/zh_CN/latest/blog/word_embedding/
import numpy as np
import pickle
import random
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, BatchNormalization
from keras.layers import Dense, LSTM, Convolution1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import Bidirectional
# 模型可视化 https://keras-cn.readthedocs.io/en/latest/other/visualization/
from keras.utils import plot_model
from IPython import display
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
# from keras.utils import multi_gpu_model
from evaluate import *


MAX_SEQUENCE_LENGTH = 200 # 句子 上限200个词
EMBEDDING_DIM = 100 # 100d 词向量


good = []
bad = []
for line in open('data/goodqueries.txt',encoding='UTF-8'):
    good.append(line.strip('\n'))
for line in open('data/badqueries.txt',encoding='UTF-8'):
    bad.append(line.strip('\n'))
print('good len:', len(good))
print('bad len:', len(bad))

data = []
labels = []

length = len(bad)
scale = 3
data.extend(good[:length * scale]) # 只取部分数据
data.extend(bad)
labels.extend([1] * length * scale)
labels.extend([0] * length)
print('data:', len(data))
print(data[0], data[-1])



# tokenizer
texts = data
tokenizer = Tokenizer(char_level=True) # 字向量
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index

# sequences
sequences = tokenizer.texts_to_sequences(data)

# padding
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

print('Found %s unique tokens.' % len(word_index))
print('Shape of data tensor:', data.shape)

token_path = 'model/tokenizer.pkl'
pickle.dump(tokenizer, open(token_path, 'wb'))


# 打乱顺序
index = [i for i in range(len(data))]
random.shuffle(index)
data = np.array(data)[index]
labels = np.array(labels)[index]

TRAIN_SPLIT = 0.8 # 20% 测试集
TRAIN_SIZE = int(len(data) * TRAIN_SPLIT)

X_train, X_test = data[0:TRAIN_SIZE], data[TRAIN_SIZE:]
Y_train, Y_test = labels[0:TRAIN_SIZE], labels[TRAIN_SIZE:]

print('train len:', len(X_train))
print('test len:', len(X_test))



import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

G = 1 # GPU 数量
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
K.set_session(session)



QA_EMBED_SIZE = 64
DROPOUT_RATE = 0.3

model = Sequential()
model.add(Embedding(len(word_index) + 1, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
model.add(Convolution1D(filters=128, kernel_size=3, padding='valid', activation='relu'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling1D(4))
model.add(Bidirectional(LSTM(QA_EMBED_SIZE, return_sequences=False, dropout=DROPOUT_RATE, recurrent_dropout=DROPOUT_RATE)))

model.add(Dense(QA_EMBED_SIZE))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(1))
model.add(BatchNormalization())
model.add(Activation("sigmoid"))

model.summary()



# pip install pydot-ng
# sudo apt-get install graphviz
# plot_model(model, to_file="img/model-cnn-blstm.png", show_shapes=True)
# display.Image('img/model-cnn-blstm.png')



EPOCHS = 3
BATCH_SIZE = 64 * G
VALIDATION_SPLIT = 0.3 # 30% 验证集

early_stopping = EarlyStopping(monitor='val_loss', patience=10)
model_checkpoint = ModelCheckpoint('model/model-cnn-blstm.h5', save_best_only=True, save_weights_only=True)
tensor_board = TensorBoard('log/tflog-cnn-blstm', write_graph=True, write_images=True)

# model = multi_gpu_model(model)

model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', precision, recall, f1])

model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
          validation_split=VALIDATION_SPLIT, shuffle=True,
          callbacks=[early_stopping, model_checkpoint, tensor_board])

print(model.evaluate(X_test, Y_test, verbose=1, batch_size=BATCH_SIZE))