import pgnToFen
import tensorflow as tf
import numpy

class PgnToTensor:
    def fill_empty_squares(self, row):
        line = []
        key = ['p','r','n','b','q','k','P','R','N','B','Q','K',]
        for i in row:
            if i.isdigit():
                line.extend([0]*int(i))
            else:
                line.append(key.index(i)+1)
        return line

    def one_hot_vector_to_tensor(self, vector):
        arr = []
        for element in vector:
            temp = tf.fill([8,8,1],element)
            arr.append(temp)

        return tf.concat([*arr],axis=2)


    def pgn_to_tensor(self, pgn):
        pgnConverter = pgnToFen.PgnToFen()
        pgnConverter.pgnToFen(map(str, pgn.split()))
        fen = pgnConverter.getFullFen().split()
        boardArray =  [self.fill_empty_squares(i) for i in fen[0].split('/')]
        boardTensor = tf.one_hot(tf.convert_to_tensor(boardArray),13,axis=2,dtype='int32')
        whoToPlay =  1 if fen[1] == 'w' else 0
        turnTensor = self.one_hot_vector_to_tensor([whoToPlay])
        castlingRight = [int(i in fen[3]) for i in ['K','Q','k','q']]
        castlingTensor = self.one_hot_vector_to_tensor(castlingRight)
        enPassant = [0 if fen[2] == '-' else ((ord(fen[2][0]) - ord('a') + 1) + (int(fen[2][1])//6)*8) == i for i in range(17)]
        enPassantTensor = self.one_hot_vector_to_tensor(enPassant)
        return tf.concat([boardTensor,turnTensor,castlingTensor,enPassantTensor], axis=2)
