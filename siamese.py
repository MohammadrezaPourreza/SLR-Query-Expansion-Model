# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Activation, Input
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D, Lambda, Concatenate
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from data_preparation import create_dataset, cosine, print_size
from tensorflow.python.framework.ops import disable_eager_execution
import tensorflow_datasets as tfds
import os

disable_eager_execution()

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# %%
X = np.load('./dataset/train/x.npy', mmap_mode='r')
Y = np.load('./dataset/train/y.npy', mmap_mode='r')
X = X[:100]
Y = Y[:100]
dataset_size = X.shape[0]
dataset = create_dataset((X,Y.argmax(axis=1)), [0,1,2])
# def convert(ds):
#     return {'input_1': ds[0], 'input_2': ds[1]}, ds[2]
# dataset = dataset.apply(convert)
print(dataset)


# %%

def build_siamese_model(inputShape, output_shape = 64):
    input = Input(shape=input_shape)
    text_layer = Conv1D(256, 5, activation='relu')(input)
    text_layer = Conv1D(256, 3, activation='relu')(text_layer)
    text_layer = MaxPooling1D(3)(text_layer)
    text_layer = Dropout(0.3)(text_layer)
    text_layer = Conv1D(256, 3, activation='relu')(text_layer)
    text_layer = MaxPooling1D(3)(text_layer)
    text_layer = Dropout(0.3)(text_layer)
    text_layer = GlobalMaxPooling1D()(text_layer)
    text_layer = Dense(256, activation='relu')(text_layer)
    output_layer = Dense(output_shape, activation='relu')(text_layer)
    model = Model(input, output_layer)
    return model
input_shape = X.shape[1:]
txtA = Input(shape=input_shape, name='input_1')
txtB = Input(shape=input_shape, name='input_2')
featureExtractor  = build_siamese_model(input_shape)
featureExtractor.summary()
featsA = featureExtractor(txtA)
featsB = featureExtractor(txtB)
distance = Lambda(cosine)([featsA, featsB])
# concat = Concatenate(axis =0)([featsA, featsB])
outputs = Dense(1, activation="sigmoid")(distance)
model = Model(name= 'model',inputs=(txtA, txtB), outputs=outputs)
model.summary()


# %%
def fit_model(model, ds, val_split = 0.1,batch_size = 64, **kwargs):
  model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])
  ds = ds.shuffle(128)
  val_size = round(val_split * dataset_size)
  val_ds = ds.take(val_size).batch(batch_size)
  train_ds = ds.skip(val_size).batch(batch_size)
  return model.fit(train_ds, validation_data=val_ds,**kwargs)


# %%

history = fit_model(model, dataset, val_split=0.2, epochs=50, batch_size=128)


# %%

model.evaluate(X_t, Y_t)


# %%
import matplotlib.pyplot as plt
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['accuracy','validation accuracy'])


# %%
from sklearn.metrics import confusion_matrix
import seaborn as sn
from sklearn.metrics import ConfusionMatrixDisplay

y_pred = model.predict(X_t)
y_pred
cm = confusion_matrix(Y_t.argmax(axis=1), y_pred.argmax(axis=1))

display_labels = ['Bad', 'Neutral', 'Good']
df_cm = pd.DataFrame(cm,index=display_labels, columns=display_labels)

# plt.figure(figsize=(10,7))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='g') # font size
plt.figure(figsize = (15,12))

# disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              # display_labels=display_labels)
# disp = disp.plot(include_values=True,
                #  cmap='viridis', ax=None, xticks_rotation='horizontal')
plt.show()
# print('recall', tp / (tp + fn))
# print('precision', tp / (tp + fp))
# print('acc', tp + tn / (tp + tn + fp + fn))


# %%
from sklearn.metrics import  classification_report


print(classification_report(Y_t.argmax(axis=1), y_pred.argmax(axis=1)))


# %%
test_df.to_csv('/content/drive/MyDrive/Uni/Thesis/query_formulation_deepnn/test_topics.csv')
train_df.to_csv('/content/drive/MyDrive/Uni/Thesis/query_formulation_deepnn/train_topics.csv')


# %%
model.save('/content/drive/MyDrive/Uni/Thesis/query_formulation_deepnn/models/simple_cnn_model')


