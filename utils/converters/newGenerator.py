import pgnToTensor

class testGenerator:
  def test_gen_data(self):
    key = {'1-0': [1,0,0] ,'0-1': [0,1,0] ,'1/2-1/2': [0,0,1]}
    file = open('../samples/lichess_2013_fens.txt','r')
    train_lines= 100000
    for _ in range(train_lines):
      x = file.readline().split() 
      yield (pgnToTensor.PgnToTensor().pgn_to_tensor((' '.join(x[:-1]))), key[x[-1]])
    file.close()

  def test_gen_val_data(self):
    key = {'1-0': [1,0,0] ,'0-1': [0,1,0] ,'1/2-1/2': [0,0,1]}
    file = open('../samples/lichess_2013_val_fens.txt','r')
    train_lines= 10000
    for _ in range(train_lines):
      x = file.readline().split() 
      yield (pgnToTensor.PgnToTensor().pgn_to_tensor((' '.join(x[:-1]))), key[x[-1]])
    file.close()