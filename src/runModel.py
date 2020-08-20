from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import createDataSet
import newGenerator
from datetime import datetime
import matplotlib.pyplot as plt


logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_delta=.0000001,verbose=1)

labeledData = tf.data.Dataset.from_generator(newGenerator.testGenerator().test_gen_data,(tf.int32,tf.int32))
labeledData = labeledData.shuffle(1000).batch(512)


valData = tf.data.Dataset.from_generator(newGenerator.testGenerator().test_gen_val_data,(tf.int32,tf.int32))
valData = valData.batch(512)


model = tf.keras.models.load_model('current_model.h5')


history = model.fit(labeledData,epochs=10,verbose=1,validation_data=valData,callbacks=[tensorboard_callback,reduce_lr])

print(history.history)

model.save('current_model.h5')

plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label = 'Val')
plt.xlabel('Epoch')
plt.ylabel('AbsoluteError')
plt.ylim([0, 1.5])
plt.legend(loc='lower right')
