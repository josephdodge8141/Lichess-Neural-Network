import pgntofen
import tensorflow as tf

class PgnToTensor:
    def fill_empty_squares(self, squares):
        line = []
        key = ['p','r','n','b','q','k','P','R','N','B','Q','K',]
        for i in squares:
            if i.isdigit():
                line.extend([0]*int(i))
            else:
                line.append(key.index(i)+1)
        return line

    def pgn_to_tensor(self, pgn):
        pgnConverter = pgntofen.PgnToFen()
        pgnConverter.pgnToFen(map(str, pgn.split()))
        fen = pgnConverter.getFullFen().split()
        board_array =  [self.fill_empty_squares(i) for i in fen[0].split('/')]
        board_tensor = tf.reshape(tf.one_hot(tf.convert_to_tensor(board_array),13,axis=0,dtype='int32'),[832])
        who_to_play =  1 if fen[1] == 'w' else 0
        turn_tensor = tf.convert_to_tensor([who_to_play],dtype='int32')
        castling_right = [int(i in fen[3]) for i in ['K','Q','k','q']]
        castling_tensor = tf.convert_to_tensor(castling_right)
        en_passant = 0 if fen[2] == '-' else (ord(fen[2][0]) - ord('a') + 1) + (int(fen[2][1])//6)*8
        en_passant_tensor = tf.one_hot(tf.convert_to_tensor(en_passant),17,dtype='int32')
        return tf.concat([board_tensor,turn_tensor,castling_tensor,en_passant_tensor],0)
