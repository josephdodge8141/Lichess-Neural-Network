import pgnToFen
import tensorflow as tf

class PgnToTensor:
    def fill_empty_squares(self, str):
        line = []
        key = ['p','r','n','b','q','k','P','R','N','B','Q','K',]
        for i in str:
            if i.isdigit():
                line.extend([0]*int(i))
            else:
                line.append(key.index(i)+1)
        return line

    def pgn_to_tensor(self, pgn):
        pgnConverter = pgnToFen.PgnToFen()
        pgnConverter.pgnToFen(map(str, pgn.split()))
        fen = pgnConverter.getFullFen().split()
        boardArray =  [self.fill_empty_squares(i) for i in fen[0].split('/')]
        boardTensor = tf.reshape(tf.one_hot(tf.convert_to_tensor(boardArray),13,axis=2,dtype='int32'),[832])
        whoToPlay =  1 if fen[1] == 'w' else 0
        turnTensor = tf.convert_to_tensor([whoToPlay],dtype='int32')
        castlingRight = [int(i in fen[3]) for i in ['K','Q','k','q']]
        castlingTensor = tf.convert_to_tensor(castlingRight)
        enPassant = 0 if fen[2] == '-' else (ord(fen[2][0]) - ord('a') + 1) + (int(fen[2][1])//6)*8
        enPassantTensor = tf.one_hot(tf.convert_to_tensor(enPassant),17,dtype='int32')
        return tf.concat([boardTensor,turnTensor,castlingTensor,enPassantTensor],0)
