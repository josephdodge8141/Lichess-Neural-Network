import tensorflow as tf
import io
import chess.pgn
import sys
import numpy
sys.path.insert(1, '../utils/converters')
import pgnToTensor



model = tf.keras.models.load_model('../models/8_19_1.h5')
white_positions = []
black_positions = []
depth = 50
board = chess.Board()


rawPgn = '1.e4 c5'
pgn = io.StringIO(rawPgn)
game = chess.pgn.read_game(pgn)
for move in game.mainline_moves():
	board.push(move)

candidates = {}
starting_num = len(board.move_stack)


for i in board.legal_moves:
	board.push(i)
	x = tf.dtypes.cast(pgnToTensor.PgnToTensor().pgn_to_tensor(str(board.fen())),tf.int32)
	x = tf.reshape(x,(1,8,8,19))
	y = tf.nn.softmax(model.predict(x)).numpy()[0]
	candidates[i] = [1,y[0]]
	white_positions.append([board.move_stack[:], y])
	board.pop()

board.reset()


for _ in range(depth):
	print(_+1)
	x = white_positions.pop(white_positions.index(max(white_positions,key=lambda x:(x[1][0])/(numpy.log(1+len(x[0])-starting_num)))))
	candidates[x[0][starting_num]][0] += 1
	candidates[x[0][starting_num]][1] += x[1][0]

	if _ == depth - 1:
		break

	for i in x[0]:
		board.push(i)
		if not board.is_valid():
			raise "Illegal Position"

	for i in board.legal_moves:
		board.push(i)
		x = tf.dtypes.cast(pgnToTensor.PgnToTensor().pgn_to_tensor(str(board.fen())),tf.int32)
		x = tf.reshape(x,(1,8,8,19))
		black_positions.append([board.move_stack[:], tf.nn.softmax(model.predict(x)).numpy()[0]])
		board.pop()

	x = black_positions.pop(black_positions.index(max(black_positions,key=lambda x:x[1][1]/(numpy.log(1+len(x[0])-starting_num)))))
	board.reset()

	for i in x[0]:
		board.push(i)
		if not board.is_valid():
			raise "Illegal Position"

	for i in board.legal_moves:
		board.push(i)
		x = tf.dtypes.cast(pgnToTensor.PgnToTensor().pgn_to_tensor(str(board.fen())),tf.int32)
		x = tf.reshape(x,(1,8,8,19))
		white_positions.append([board.move_stack[:], tf.nn.softmax(model.predict(x)).numpy()[0]])
		board.pop()
	board.reset()
for key in candidates:
	candidates[key][1] /= candidates[key][0]
print(candidates)