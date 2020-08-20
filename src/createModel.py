import tensorflow as tf

initializer = tf.keras.initializers.HeNormal()
regularizer = tf.keras.regularizers.L1(l1=.001)
inputs = tf.keras.Input(shape=(8,8,19))
filters = 32

x = tf.keras.layers.Conv2D(filters,(3,3),padding='same',kernel_regularizer=regularizer, bias_regularizer=regularizer, kernel_initializer=initializer)(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)

x = tf.keras.layers.Conv2D(filters,(3,3),padding='same',kernel_regularizer=regularizer, bias_regularizer=regularizer, kernel_initializer=initializer)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)

x = tf.keras.layers.Conv2D(filters,(3,3),padding='same',kernel_regularizer=regularizer, bias_regularizer=regularizer, kernel_initializer=initializer)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)

x = tf.keras.layers.Conv2D(1,(1,1),padding='same',kernel_regularizer=regularizer, bias_regularizer=regularizer, kernel_initializer=initializer)(x)
x = tf.keras.layers.Flatten()(x)

outputs = tf.keras.layers.Dense(3,kernel_regularizer=regularizer, bias_regularizer=regularizer, kernel_initializer=initializer)(x)

model = tf.keras.Model(inputs=inputs,outputs=outputs)

model.compile(optimizer=tf.keras.optimizers.Adadelta(learning_rate=.1),loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=[tf.keras.metrics.CategoricalAccuracy()])
model.summary()
model.save('current_model.h5')
