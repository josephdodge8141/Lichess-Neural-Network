import tensorflow as tf
import sys
sys.path.insert(1, '../utils/converters')
import pgnToTensor


model = tf.keras.models.load_model('../models/8_15.h5')


pos = {'White win': '4k3/8/8/3PP3/2P5/BPN2NP1/P3QPBP/3RR1K1 w - - 0 1', 
'Black win':'2rr2k1/4q2p/2n1bnpb/p4p2/1p2p3/2pp4/8/4K3 b - - 0 1',
'Mainline Najdorf': 'r1b1k2r/2qnbppp/p2ppn2/1p4B1/3NPPP1/2N2Q2/PPP4P/2KR1B1R w kq - 0 11',
'Fried Liver': 'r1bq1b1r/ppp3pp/2n1k3/3np3/2B5/5Q2/PPPP1PPP/RNB1K2R w KQ - 2 8',
'Modern with f5': 'rnbq1rk1/ppp3bp/3p2p1/3Ppp1n/2P1P3/2N1BN2/PP2BPPP/R2QK2R w KQ - 0 9',
'Italian': 'r2qk2r/ppp2ppp/2np1n2/2b1p3/2B1P1b1/2PP1N2/PP3PPP/RNBQ1RK1 w kq - 2 7',
'London': 'r1bq1rk1/pp2bpp1/2n1pn1p/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2Q1RK1 w - - 2 9',
'Nimzo': 'rnbq1rk1/pppp1ppp/4pn2/8/1bPP4/2N1P3/PP3PPP/R1BQKBNR w KQ - 1 5',
'Rook and King': '2r5/8/3k4/8/8/4K3/8/5R2 w - - 0 1',
'Equal King and Pawn': '8/pp4k1/5ppp/8/8/5PPP/PP4K1/8 w - - 0 1',
'Winning king and Pawn': '2k5/8/4K2P/8/8/p7/8/8 w - - 0 1',
'1.e4':'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1'}

for item in pos:
  x = tf.dtypes.cast(pgnToTensor.PgnToTensor().pgn_to_tensor(str(pos[item])),tf.int32)
  x = tf.reshape(x,(1,8,8,19))
  print(tf.nn.softmax(model.predict(x)).numpy() ,item)
	
