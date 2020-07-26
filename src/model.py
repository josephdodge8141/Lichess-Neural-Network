import tensorflow as tf
import createDataSet


#creates a dataset of position tensors and a dataset of result labels

labeledData = tf.data.Dataset.from_generator(createDataSet.Generator().gen_data,(tf.string,tf.uint8))
labeledData = labeledData.map(createDataSet.Generator().converter)
labeledData = labeledData.shuffle(100000, reshuffle_each_iteration=True).batch(512)

valData = tf.data.Dataset.from_generator(createDataSet.Generator().gen_val_data,(tf.string,tf.uint8))
valData = valData.map(createDataSet.Generator().converter).shuffle(100000).batch(512)


print('Finished loading data')


inputs = tf.keras.Input(shape=(8,8,14))

x = tf.keras.layers.Conv2D(64,(3,3),padding='same',activity_regularizer='l2',bias_regularizer='l2')(inputs)
x = tf.keras.layers.BatchNormalization()(x)
y = tf.keras.layers.ReLU()(x)

x = tf.keras.layers.Conv2D(64,(3,3),padding='same',activity_regularizer='l2',bias_regularizer='l2')(y)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.Conv2D(64,(3,3),padding='same',activity_regularizer='l2',bias_regularizer='l2')(x)
x = tf.keras.layers.BatchNormalization()(x)
x += y
y = tf.keras.layers.ReLU()(x)

x = tf.keras.layers.Conv2D(64,(3,3),padding='same',activity_regularizer='l2',bias_regularizer='l2')(y)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.Conv2D(64,(3,3),padding='same',activity_regularizer='l2',bias_regularizer='l2')(x)
x = tf.keras.layers.BatchNormalization()(x)
x += y
y = tf.keras.layers.ReLU()(x)


x = tf.keras.layers.Conv2D(64,(3,3),padding='same',activity_regularizer='l2',bias_regularizer='l2')(y)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.Conv2D(64,(3,3),padding='same',activity_regularizer='l2',bias_regularizer='l2')(x)
x = tf.keras.layers.BatchNormalization()(x)
x += y
y = tf.keras.layers.ReLU()(x)

x = tf.keras.layers.Conv2D(1,(1,1))(y)
x = tf.keras.layers.BatchNormalization()(x)
y = tf.keras.layers.ReLU()(x)


x = tf.keras.layers.Dense(64,activation='relu')(y)
outputs = tf.keras.layers.Dense(1,activation='tanh')(x)

model = tf.keras.Model(inputs=inputs,outputs=outputs)
model.summary()


model.compile(optimizer=tf.keras.optimizers.Adam(),loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.MeanAbsoluteError(),tf.keras.metrics.RootMeanSquaredError()])

history = model.fit(labeledData,epochs=3,verbose=1,validation_data=valData)

print(history.history)

