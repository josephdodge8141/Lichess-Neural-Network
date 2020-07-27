import tensorflow as tf
import createDataSet


#creates a dataset of position tensors and a dataset of result labels

labeledData = tf.data.Dataset.from_generator(createDataSet.Generator().gen_data,(tf.string,tf.int32))
labeledData = labeledData.map(createDataSet.Generator().converter)
labeledData = labeledData.shuffle(10000, reshuffle_each_iteration=True).batch(512)

valData = tf.data.Dataset.from_generator(createDataSet.Generator().gen_val_data,(tf.string,tf.int32))
valData = valData.map(createDataSet.Generator().converter).shuffle(10000).batch(512)

print('Finished loading data')


inputs = tf.keras.Input(shape=(8,8,35))

a = tf.keras.layers.Conv2D(16,(3,3))(inputs)
a = tf.keras.layers.BatchNormalization()(a)
a = tf.keras.layers.ReLU()(a)

b = tf.keras.layers.Conv2D(16,(3,3))(inputs)
b = tf.keras.layers.BatchNormalization()(b)
b = tf.keras.layers.ReLU()(b)

a1= tf.keras.layers.Conv2D(16,(3,3))(a)
a1 = tf.keras.layers.BatchNormalization()(a1)
a1 = tf.keras.layers.ReLU()(a1)

a2 = tf.keras.layers.Conv2D(16,(3,3))(a)
a2 = tf.keras.layers.BatchNormalization()(a2)
a2 = tf.keras.layers.ReLU()(a2)

b1 = tf.keras.layers.Conv2D(16,(3,3))(b)
b1 = tf.keras.layers.BatchNormalization()(b1)
b1 = tf.keras.layers.ReLU()(b1)

b2 = tf.keras.layers.Conv2D(16,(3,3))(b)
b2 = tf.keras.layers.BatchNormalization()(b2)
b2 = tf.keras.layers.ReLU()(b2)

x = tf.keras.layers.Concatenate(axis=-3)([a1,b1])
y = tf.keras.layers.Concatenate(axis=-3)([b2,a2])
z = tf.keras.layers.Concatenate(axis=-2)([x,y])



a = tf.keras.layers.Conv2D(16,(3,3))(z)
a = tf.keras.layers.BatchNormalization()(a)
a = tf.keras.layers.ReLU()(a)

b = tf.keras.layers.Conv2D(16,(3,3))(z)
b = tf.keras.layers.BatchNormalization()(b)
b = tf.keras.layers.ReLU()(b)

a1= tf.keras.layers.Conv2D(16,(3,3))(a)
a1 = tf.keras.layers.BatchNormalization()(a1)
a1 = tf.keras.layers.ReLU()(a1)

a2 = tf.keras.layers.Conv2D(16,(3,3))(a)
a2 = tf.keras.layers.BatchNormalization()(a2)
a2 = tf.keras.layers.ReLU()(a2)

b1 = tf.keras.layers.Conv2D(16,(3,3))(b)
b1 = tf.keras.layers.BatchNormalization()(b1)
b1 = tf.keras.layers.ReLU()(b1)

b2 = tf.keras.layers.Conv2D(16,(3,3))(b)
b2 = tf.keras.layers.BatchNormalization()(b2)
b2 = tf.keras.layers.ReLU()(b2)

x = tf.keras.layers.Concatenate(axis=-3)([a1,b1])
y = tf.keras.layers.Concatenate(axis=-3)([b2,a2])
z = tf.keras.layers.Concatenate(axis=-2)([x,y])


x = tf.keras.layers.Flatten()(z)
outputs = tf.keras.layers.Dense(1,activation='tanh')(x)

model = tf.keras.Model(inputs=inputs,outputs=outputs)
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(),loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.MeanAbsoluteError(),tf.keras.metrics.RootMeanSquaredError()])

history = model.fit(labeledData,epochs=3,verbose=1,validation_data=valData)

print(history.history)
