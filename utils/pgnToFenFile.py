import chess.pgn
import io

raw = open('../samples/lichess_db_standard_rated_2013-01.pgn','r')
clean = open('../samples/lichess_2013_test.txt','w')

def flop_fen(fen):
	fen = fen.split()
	key = {'b':'w', 'w':'b', '3':'6', '6':'3'}
	fen[0] = '/'.join(fen[0].swapcase().split('/')[::-1])
	fen[1] = key[fen[1]]
	fen[2] = fen[2].swapcase()
	if fen[3] != '-':
		fen[3] = fen[3][0] + key[fen[3][1]]
	return ' '.join(fen)

def pgn_to_fen(rawPgn):
	pgn = io.StringIO(rawPgn)
	game = chess.pgn.read_game(pgn)
	board = game.board()
	for move in game.mainline_moves():
		board.push(move)
	return board.fen()

for _ in range(2):
	key = {'1-0': [1,0,0] ,'0-1': [0,1,0],'1/2-1/2': [0,0,1]}
	line = raw.readline().split
	while line != []:
		line = raw.readline().split()
	y = raw.readline().split()
	while len(y) > 0:
		line.extend(y)
		y = raw.readline().split()
	if len(line) > 30:
		whole_moves = (len(line)-1)//3
		for i in range(whole_moves):
			fen = pgn_to_fen(' '.join(line[:i*3+2]))
			label = key[line[-1]]
			clean.write(fen + ' ' + str(key[line[-1]]) + '\n')
			clean.write(flop_fen(fen) + ' ' + str(key[line[-1]]) + '\n')
			fen = pgn_to_fen(' '.join(line[:i*3+3]))
			clean.write(fen + ' ' + str(key[line[-1]]) + '\n')
			clean.write(flop_fen(fen) + ' ' + str(key[line[-1]]) + '\n')
clean.close()

