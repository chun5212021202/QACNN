import glob
import re
from nltk.stem.snowball import SnowballStemmer
import json
import numpy as np


snowball = SnowballStemmer("english")
re_alphanumeric = re.compile('[^a-z0-9 -]+')
re_multispace = re.compile(' +')



def processLine(line):
	
	line = line.lower()
	line = re_alphanumeric.sub('', line)
	line = re_multispace.sub(' ', line)
	
	return line

def processWord(word):
	word = word.lower()
	word = re_alphanumeric.sub('', word)
	word = re_multispace.sub(' ', word)
	word = snowball.stem(word)
	return word

def listSplitPlot(fd):
	lines = []
	while True:
		line = fd.readline()
		if line == '':
			break
		line = line.strip('\r\n')
		if len(line) > 1:
			line = processLine(line)
			line = line.split()
			lines.append(line)
	return lines


def compute():
	a=0
	b=0
	register = {}
	fNameList = glob.glob('../raw_data/plot/*.wiki')
	
	with open('../word_vec/glove.42B.300d.json') as glove_file:
		glove_dict = json.load(glove_file)
		for fName in fNameList:
			
			fd = open(fName, 'r')
			docName = fName.split('/')[-1]
			wordList = listSplitPlot(fd)
			fd.close()
			plot_vec = []
			for sentence in wordList:
				sentence_vec = []
				for word in sentence:
					a+=1
					word_stem = snowball.stem(word)
					if word in glove_dict.keys():
						word_vec = glove_dict[word]
						sentence_vec.append(word_vec)
						b+=1

					elif word_stem in glove_dict.keys():
						word_vec = glove_dict[word_stem]
						sentence_vec.append(word_vec)
						b+=1
						
					elif word in register.keys():
						word_vec = register[word]
						sentence_vec.append(word_vec) 
						print(word)

					else:
						word_vec = (0.5*np.random.randn(300)).tolist()
						register[word] = word_vec
						sentence_vec.append(word_vec)
						print(word)
				plot_vec.append(sentence_vec)
			json.dump(plot_vec, open('../output_data/plot/'+docName+'.json','w'))
	json.dump(register, open('../word_vec/register.json','w'))
	print('all: ',a,'  glove: ',b)


compute()
