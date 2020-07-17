import pgntofen
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
def fill_empty_squares(str):
    line = []
    key = ['p','r','n','b','q','k','P','R','N','B','Q','K',]
    for i in str:
        if i.isdigit():
            line.extend([0]*int(i))
        else:
            line.append(key.index(i)+1)
    return line
pgnConverter = pgntofen.PgnToFen()
PGNMoves = 'b4 a5'
pgnConverter.pgnToFen(map(str, PGNMoves.split()))
fen = pgnConverter.getFullFen()
def fen_to_tensor(fen):
    x = fen.split()
    board_array =  [fill_empty_squares(i) for i in x[0].split('/')]
    board_tensor = tf.reshape(tf.one_hot(tf.convert_to_tensor(board_array),13,axis=0,dtype='int32'),[832])
    who_to_play =  1 if x[1] == 'w' else 0
    turn_tensor = tf.convert_to_tensor([who_to_play],dtype='int32')
    castling_right = [int(i in x[3]) for i in ['K','Q','k','q']]
    castling_tensor = tf.convert_to_tensor(castling_right)
    en_passant = 0 if x[2] == '-' else (ord(x[2][0]) - ord('a') + 1) + (int(x[2][1])//6)*8
    en_passant_tensor = tf.one_hot(tf.convert_to_tensor(en_passant),17,dtype='int32')

    return tf.concat([board_tensor,turn_tensor,castling_tensor,en_passant_tensor],0)

sess = tf.Session()
print(sess.run(fen_to_tensor(fen)))
#This is a lichess neural network
#this is for nn arch