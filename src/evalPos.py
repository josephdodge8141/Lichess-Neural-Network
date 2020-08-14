import tensorflow as tf
import sys
sys.path.insert(1, '../utils/converters')
import pgnToTensor


model = tf.keras.models.load_model('current_model.h5')

pos = {'W' : '1. e3 Nf6 2. Qf3 Ng8 3. Qxb7 Nf6 4. Qxb8 Ng8 5. Qxa8 Nf6 6. Qxa7 Ng8 7. Qxc7 Nf6 8. Qxc8 Ng8 9. Nc3 Nf6 10. Nf3 Ng8 11. d4 Nf6 12. Bd3 e6 13. Bd2 Be7 14. O-O-O O-O 15. Qxd8 h6 16. Qxe7 d6',
'B' : '1. Nf3 e6 2. Ng1 Qf6 3. Nf3 Qxb2 4. Ng1 Qxa1 5. Nf3 Qxb1 6. Ng1 Qxc1 7. e3 Nf6 8. Be2 Nc6 9. Nf3 d5 10. O-O Bd7 11. Ne1 Qxd1 12. Nf3 Qxe2 13. Ne1 O-O-O 14. Nf3 Bd6 15. Ne1 Qxd2 16. Nf3 Qxc2 17. Ne1 Qxa2 18. Nf3',
'2NA' : '1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 4. Ng5',
'London' : '1. d4 d5 2. Bf4 e6 3. Nf3 Nf6 4. e3 c5 5. c3 Be7 6. Nbd2 O-O 7. Bd3 Nc6 8. O-O',
'c4' : '1.c4', 
'd4' : '1.d4', 
'e4' : '1.e4', 
'f4' : '1.f4'}

for item in pos:
  x = tf.dtypes.cast(pgnToTensor.PgnToTensor().pgn_to_tensor(str(pos[item])),tf.int32)
  x = tf.reshape(x,(1,8,8,19))
  print(item, tf.nn.softmax(model.predict(x)).numpy())
	
