import tensorflow as tf
import tensorflow_datasets as tfds
import fenToTensor
import pgntofen 
import createDataSet

#creates a dataset of position tensors and a dataset of result labels
labeledData = tf.data.Dataset.from_generator(createDataSet.genData,(tf.string,tf.uint8))
labeledData = labeledData.map(createDataSet.converter)
labeledData = labeledData.shuffle(100000, reshuffle_each_iteration=True).batch(512)

valData = tf.data.Dataset.from_generator(createDataSet.genValData,(tf.string,tf.uint8))
valData = valData.map(createDataSet.converter).shuffle(100000).batch(512)


print('Finished loading data')


model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(854,)))
model.add(tf.keras.layers.Reshape((14,61,1)))
model.add(tf.keras.layers.Conv2D(64,(3,3)))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Reshape((24,29,16)))
model.add(tf.keras.layers.Conv2D(64,(3,3)))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Conv2D(64,(3,3)))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Reshape((8,10,16)))
model.add(tf.keras.layers.Conv2D(64,(3,3)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(3))
model.summary()


model.compile(optimizer=tf.keras.optimizers.Adam(),loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=[tf.keras.metrics.TopKCategoricalAccuracy(1)])

history = model.fit(labeledData,epochs=3,verbose=1,validation_data=valData)

print(history.history)

