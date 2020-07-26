import tensorflow as tf
import createDataSet


#creates a dataset of position tensors and a dataset of result labels
labeledData = tf.data.Dataset.from_generator(createDataSet.genData,(tf.string,tf.uint8))
labeledData = labeledData.map(createDataSet.converter)
labeledData = labeledData.shuffle(100000, reshuffle_each_iteration=True).batch(512)

valData = tf.data.Dataset.from_generator(createDataSet.genValData,(tf.string,tf.uint8))
valData = valData.map(createDataSet.converter).shuffle(100000).batch(512)


print('Finished loading data')


inputs = tf.keras.Input(shape=(8,8,14))
model.add(tf.keras.layers.Conv2D(64,(3,3),padding='same',activity_regularizer='l2',bias_regularizer='l2',activation='relu'))
model.add(tf.keras.layers.Conv2D(64,(3,3),padding='same',activity_regularizer='l2',bias_regularizer='l2',activation='relu'))
model.add(tf.keras.layers.Conv2D(64,(3,3),padding='same',activity_regularizer='l2',bias_regularizer='l2',activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
model.summary()


model.compile(optimizer=tf.keras.optimizers.Adam(),loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.MeanAbsoluteError(),tf.keras.metrics.RootMeanSquaredError()])

history = model.fit(labeledData,epochs=3,verbose=1,validation_data=valData)

print(history.history)

