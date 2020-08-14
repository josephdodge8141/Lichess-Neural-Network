import tensorflow as tf
import numpy
import io
import chess.pgn
#this class is set up for the chess library on google colab, which is not up to date. To run on the current chess library,
#change game.main_line in pgn_to_fen to game.mainline_moves
class PgnToTensor:

    def pgn_to_fen(self, rawPgn):
      pgn = io.StringIO(rawPgn)
      game = chess.pgn.read_game(pgn)
      board = game.board()
      for move in game.mainline_moves():
        board.push(move)
      return board.fen()

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
        if len(vector) > 1:
            for element in vector:
                temp = tf.fill([8,8,1],element)
                arr.append(temp)
        else:
            temp = tf.fill([8,8],vector[0])
            arr.append(temp)
        return tf.concat([*arr],axis=2)

    def en_passant_helper(self, square):
      relv = [[0]*8,[0]*8]
      relv[square[0]][square[1]] = 1
      arr = [[0]*8,relv[1],*[[0]*8]*4,relv[0],[0]*8]
      return tf.convert_to_tensor(arr)

    def pgn_to_tensor(self, pgn):
        fen = pgn.split()
        boardArray =  [self.fill_empty_squares(i) for i in fen[0].split('/')]
        boardTensor = tf.one_hot(tf.convert_to_tensor(boardArray),13,axis=2,dtype='int32')
        whoToPlay =  1 if fen[1] == 'w' else 0
        turnTensor = self.one_hot_vector_to_tensor([whoToPlay])
        castlingRight = [int(i in fen[2]) for i in ['K','Q','k','q']]
        castlingTensor = self.one_hot_vector_to_tensor(castlingRight)
        if fen[3] == '-':
            enPassantTensor = tf.convert_to_tensor([*[[0]*8]*8])
        else:
            enPassantTensor = self.en_passant_helper([int(fen[2][1])//6, ord(fen[2][0]) - ord('a')])
        return tf.concat([boardTensor,castlingTensor,tf.stack([turnTensor,enPassantTensor],axis=2)], axis=2)
