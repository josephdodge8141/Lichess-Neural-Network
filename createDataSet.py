import fenToTensor
import pgntofen 
import tensorflow as tf

def genData(): #generates usable position pngs from a file with raw pngs
	f = open('oldPngs.txt')
	for _ in range(1000):
		try:
			line = f.readline().split
			while line != []:
				line = f.readline().split()
			y = f.readline().split()
			while len(y) > 0:
				line.extend(y)
				y = f.readline().split()
			key = {'1-0': [1,0,0] ,'0-1': [0,0,1] ,'1/2-1/2': [0,1,0]}
			if len(line) > 10 and line[-1] in key:
				for i in range((len(line)-1)//2):#This range comes from each full turn being comprised of 
					yield (' '.join(line[:i*2+1]),key[line[-1]]) #the number of the turn, whites move, and blacks move. Subract 1 because the score at the end
					yield (' '.join(line[:i*2+2]),key[line[-1]]) #From the full pgn x, each sub position is yielded
		except StopIteration:
			break

def genValData(): #generates usable position pngs from a file with raw pngs
	f = open('wccPngs.txt')
	for _ in range(100):
		try:
			line = f.readline().split
			while line != []:
				line = f.readline().split()
			y = f.readline().split()
			while len(y) > 0:
				line.extend(y)
				y = f.readline().split()
			key = {'1-0': [1,0,0] ,'0-1': [0,0,1] ,'1/2-1/2': [0,1,0]}
			if len(line) > 10 and line[-1] in key:
				for i in range((len(line)-1)//2):#This range comes from each full turn being comprised of 
					yield (' '.join(line[:i*2+1]),key[line[-1]]) #the number of the turn, whites move, and blacks move. Subract 1 because the score at the end
					yield (' '.join(line[:i*2+2]),key[line[-1]]) #From the full pgn x, each sub position is yielded
		except StopIteration:
			break

def converter(pos,label):
	x =tf.dtypes.cast(fenToTensor.pgnToTensor(str(pos)),tf.uint8)
	y = tf.convert_to_tensor(label)
	return (x,y)
