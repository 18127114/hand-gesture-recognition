import numpy as np
import time
from nltk import CFG
from nltk.parse.generate import generate, demo_grammar

def MED(string1, string2):	
	l1 = len(string1)
	l2 = len(string2)

	d = [[0 for i in range(l2+1)] for j in range(l1+1)]
	for i in range(l1+1):
		d[i][0] = i
	for j in range(l2+1):
		d[0][j] = j
	for i in range(1,l1+1):
		for j in range(1,l2+1):
			if (string1[i-1] == string2[j-1]):
				d[i][j] = d[i-1][j -1]
			else:
				d[i][j] = np.min(np.array([d[i-1][j], d[i][j-1], d[i-1][j -1]])) + 1

	return d[l1][l2]


def readlineFile(filename):
    with open(filename, "r", encoding = "utf8") as file:
        data = file.readlines()
        file.close() 
       
    return data

def dictionary(VDic):
	dic = list()
	posTag = list()

	for vob in VDic:
		pos = vob.find('\t',0)
		dic.append(vob[:pos])
		start = pos + 2
		end = vob.find('\n',0)
		posTag.append(vob[start:end])

	return dic, posTag

data  = readlineFile("VDic_uni.txt")
dic, posTag = dictionary(data)

def similar(word, dictionary, posTag):
	size = len(dictionary)
	a = np.zeros(size)
	start = time.time()
	for i in range(size):
		a[i] = MED(word,dictionary[i]) 

	
	end = time.time()
	# print(end-start)

	indexMinValue = np.where(a == np.amin(a))

	return list(dictionary[i] for i in indexMinValue[0]), list(posTag[i] for i in indexMinValue[0])

# for sentence in generate(grammar, n = 10):
# 	print(' '.join(sentence))
def addRule(rule, related, posTag):
	for i in range(len(posTag)):
		posTag[i] = posTag[i].replace(', ',' ')
		splitTag = posTag[i].split(' ')
		for j in range(len(splitTag)):
			# print(splitTag[j])
			_rule = splitTag[j] + ' -> ' + '"' + related[i] + '"'
			rule = rule + "\n" + _rule

	return rule
def checkSpelling(text, vocabulary, posTag):
	text = text.lower()
	splitText = text.split(' ')
	result = ""
	flag = 0

	rule = "\n" + "S -> NP VP" + "\n" + "VP -> V N | V NP PP" + "\n" + "NP -> N"
	for i in range(len(splitText)):
		if splitText[i] not in vocabulary:
			flag = 1
			voc, tag = similar(splitText[i], vocabulary, posTag)
			rule = addRule(rule, voc, tag)
			# grammar = CFG.fromstring(rule)
			# text = text.replace()
			# print(voc,tag)
		
	if flag == 0:
		result = text.upper()
		print(result)
	else:
		grammar = CFG.fromstring(rule)
		for sentence in generate(grammar, n = 30):
			print(' '.join(sentence))
			# print(sentence[])
			print(sentence[2])
	# print(flag)
	print(rule)
	return text 
