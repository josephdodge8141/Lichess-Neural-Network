
##### PgnToFen

You can insert one and one move if you want.

#### E.g
```python
import pgntofen # assumes you have pgntofen.py in the same directory, or you know how to handle python modules.
pgnConverter = pgntofen.PgnToFen()
pgnConverter.move('d4')
fen = pgnConverter.getFullFen()
#fen will be 'rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b - KQkq'
```

### Moves

You can send a string, or an array of strings.
If you send a string, it may be a valid PGN Line (`1.e4 d5 2.Nf3 ....`)
if it'a and array of strings, you may only send the actulle moves (`['e4', 'd5', 'Nf3']`)

#### E.g

```python
import pgntofen # assumes you have pgntofen.py in the same directory, or you know how to handle python modules.
pgnConverter = pgntofen.PgnToFen()
PGNMoves = 'd4 d5'
pgnConverter.pgnToFen(PGNMoves.split(''))
fen = pgnConverter.getFullFen()
#fen will be 'rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR - KQkq'
```

### pgnFile

parse a pgnFile that may have sveral pgn games.

#### E.g

```python
pgnConverter = pgntofen.PgnToFen()
pgnConverter.resetBoard()
file = "test/Example.pgn"
stats =  pgnConverter.pgnFile(file);
# stats => {
# 'failed': [<pgntofen-error-obj>, ...],
# 'succeeded': [<game-obj>, ...]
# }

# a game-obj: (game_info, fens)
# pgntofen-error-obj: (game_info, lastMove, fen, error)
# fens: array of fen
# game_info is all the line in the pgn file working as a header before the game (e.g: all lines with [...])
```

PgnToTensor

This will accept any pgn and print out a 1 hot tensor which includes information about the position, whose turn it is, castling rights, and legal enpassants.

### E.g

```python

import pgnToTensor # assumes you have pgntofen.py in the same directory, or you know how to handle python modules.
pgnConverter = pgnToTensor.PgnToTensor()
PGNMoves = 'd4 d5'
tensor = pgnConverter.pgnToTensor(PGNMoves)
print(tensor)

# Result will be

#tf.Tensor(
#[0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 0 1
# 1 1 1 1 1 1 1 1 1 1 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1
# 1 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
# 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0
# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0
# 0 0 0], shape=(854,), dtype=int32)
```