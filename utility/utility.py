import numpy as np
from numpy import sum
from random import shuffle
import operator
import json
import csv
from pprint import pprint


def batchPadding(batchSentence,max_len): 
	""" Padding batch of question or choices
		Parameters
		----------
			batchSentence : 3d numpy array, 3rd level list
				each dimension represents (batch, sentence, word_vec) respectively
			max_len : int
				length of the longest sentence (default is 50)
		Returns
		-------
			3d numpy array
				batch of question or choices (padded)
	"""
	newBatchSentence = []
	filler = [0]*300
	for sen in batchSentence:
		newBatchSentence.append(sen + [filler]*(max_len-len(sen)))
	return np.asarray(newBatchSentence)



def varSentencePadding(plots,max_len):
	""" Padding batch of plots
		Parameters
		----------
			plots : 4d numpy array, 4th level list
				each dimension represents (batch, plot, sentence, word_vec) respectively
			max_len : int
				length of the longest sentence (default is 100)
		Returns
		-------
			4d numpy array
				batch of plots (padded)
	"""
	newPlot = []

	filler = np.zeros(300)
	max_plot_len = 0
	plot_filler = np.zeros((max_len,300))
	max_plot_len = 101
	for plot in plots:
		newSentence = []
		for sentences in plot:
			newSentence.append(sentences+[filler]*(max_len-len(sentences)))

		newPlot.append(newSentence+[plot_filler]*(max_plot_len-len(plot)))

	return np.asarray(newPlot)


def emsenble_test(ACMNet_object,plotFilePath,data,batchSize,max_len,choice,model_file_list,isVal):
	""" test the accuracy of given data (emsenble several models)
		Parameters
		----------
			ACMNet_object : object instance
				from class "MODEL" built up by tensorflow

			plotFilePath : string
				path of plot files in word vector form

			data : list of dictionary
				list of training data, each of which has structure of :
					{
						'imdb_key' : string
						'question' : 2d numpy array (sentence, word_vec)
						'answers' : 3d numpy array (5, sentence, word_vec)
						'correct_index' : int
					}

			batchSize : int

			max_len : list
				only 3 elements are in the list, 
				["max length of sentence in plot", "max length of question", "max length of choice"]

			choice : int
				number of given choices (default 5)

			model_file_list : list of string
				all the parameter file path (from tensorflow)

			isVal : bool
				whether the given data is validation data
				validation data : true
				training data : false
		Returns
		-------
			float
				accuracy of given data
	"""    
	count_data = 0
	for i in data:
	   count_data+=1    
	predict_total_list = np.zeros((count_data,choice))
	test_csv = []
	batchQuestionId = []
	for model_file_ind,model_file in enumerate(model_file_list):        
		batchP = []
		batchQ = []
		batchAnsVec = []
		batchAnsOpt = []
 
		predictAns = []
		filler = [0]*300
		ACMNet_object.restore(model_file) 
		break
		for question in data:
			imdb_key = plotFilePath+question["imdb_key"]+".split.wiki.json"
			with open(imdb_key) as data_file:
					testP = json.load(data_file)
			batchP.append(testP)
			batchQ.append(question["question"])
			AnsOption = []
			
			for j in range(choice):
					if question["answers"][j]:
						pass
					else:
						question["answers"][j] = [filler]
					batchAnsVec.append(question["answers"][j])
					AnsOption.append(0)
			AnsOption[question["correct_index"]] = 1
			batchAnsOpt.append(AnsOption)
			if model_file_ind == 0:
					batchQuestionId.append(question["qid"])
			if len(batchP) == batchSize:

					batchP = varSentencePadding(batchP,max_len[0])
					batchQ = batchPadding(batchQ,max_len[1])
					batchAnsVec =  batchPadding(batchAnsVec,max_len[2])
			  
					tmp = ACMNet_object.predict_prob(batchP,batchQ,batchAnsVec)

					predictAns.extend(tmp.tolist())
					print (len(predictAns))
					batchP = []
					batchQ = []
					batchAnsVec = []  
		predict_total_list+=np.asarray(predictAns)
	predict_ind = np.argmax(predict_total_list,axis=1).tolist()

	correct_num = 0.0
	question_num = len(predict_total_list)
	
	for idx in range(question_num):
			predictIdx = predict_ind[idx]
			test_csv.extend([[batchQuestionId[idx],predictIdx]])
			if batchAnsOpt[idx][predictIdx] == 1:
					correct_num += 1
	print ('correct:%d total:%d rate:%f' % (correct_num,question_num,correct_num/question_num))
	f = open("val_logit_emsemble.csv","w")
	w = csv.writer(f)
	w.writerows(test_csv)
	f.close()
	return correct_num/question_num

def test(ACMNet_object,plotFilePath,data,batchSize,max_len,choice,isVal):
	""" test the accuracy of given data
		Parameters
		----------
			ACMNet_object : object instance
				from class "MODEL" built up by tensorflow

			plotFilePath : string
				path of plot files in word vector form

			data : list of dictionary
				list of training data, each of which has structure of :
					{
						'imdb_key' : string
						'question' : 2d numpy array (sentence, word_vec)
						'answers' : 3d numpy array (5, sentence, word_vec)
						'correct_index' : int
					}
			batchSize : int

			max_len : list
				only 3 elements are in the list, 
				["max length of sentence in plot", "max length of question", "max length of choice"]

			choice : int
				number of given choices (default 5)

			isVal : bool
				whether the given data is validation data
				validation data : true
				training data : false
		Returns
		-------
			float
				accuracy of given data
	"""    
	batchP = []
	batchQ = []
	batchAnsVec = []
	batchAnsOpt = []

	###    Test validation set   #########
	predictAns = []
	filler = [0]*300
	for question in data:
			imdb_key = plotFilePath+question["imdb_key"]+".split.wiki.json"
			with open(imdb_key) as data_file:
					testP = json.load(data_file)
			batchP.append(testP)
			batchQ.append(question["question"])
			AnsOption = []

			for j in range(choice):
					if question["answers"][j]:
						pass
					else:
						question["answers"][j] = [filler]
					batchAnsVec.append(question["answers"][j])
					AnsOption.append(0)
			AnsOption[question["correct_index"]] = 1
			batchAnsOpt.append(AnsOption)
			
			if len(batchP) == batchSize:

					batchP = varSentencePadding(batchP,max_len[0])
					batchQ = batchPadding(batchQ,max_len[1])
					batchAnsVec =  batchPadding(batchAnsVec,max_len[2])

					tmp = ACMNet_object.predict(batchP,batchQ,batchAnsVec)

					predictAns.extend(tmp)
					batchP = []
					batchQ = []
					batchAnsVec = [] 
   

	###calculate accuracy###
	correct_num = 0.0
	question_num = len(predictAns)
	
	for idx in range(question_num):
			predictIdx = predictAns[idx]
			if batchAnsOpt[idx][predictIdx] == 1:
					correct_num += 1
	print ('correct:%d total:%d rate:%f' % (correct_num,question_num,correct_num/question_num))
	print(predictAns)
	return correct_num/question_num

def Train_Test(train_q,val_q,plotFilePath,ACMNet_object,epoch,lr,batchSize,dropoutRate,choice,max_len) :
	""" train for one epoch
		Parameters
		----------
			train_q, val_q : list of dictionary
				list of training data, each of which has structure of :
					{
						'imdb_key' : string
						'question' : 2d numpy array (sentence, word_vec)
						'answers' : 3d numpy array (5, sentence, word_vec)
						'correct_index' : int
					}

			plotFilePath : string
				path of plot files in word vector form

			ACMNet_object : object instance
				from class "MODEL" built up by tensorflow

			epoch : int

			lr : float
				learning rate

			batchSize : int

			dropoutRate : float

			choice : int
				number of given choices (default 5)

			max_len : list
				only 3 elements are in the list, 
				["max length of sentence in plot", "max length of question", "max length of choice"]
	"""

		accuracy_list = {'train':[],'val':[]}
		global_step = 0
		filler = [0]*300        
			
		for i in range(epoch):      

				batchP = []
				batchQ = []
				batchAnsVec = []
				batchAnsOpt = []
				batchcount_epoch = 0
				totalcost_epoch = 0
				for question in train_q:
						imdb_key = plotFilePath+question["imdb_key"]+".split.wiki.json"
						with open(imdb_key) as data_file:
							P = json.load(data_file)
						batchP.append(P) #need check
						batchQ.append(question["question"])
						AnsOption = []

						for j in range(choice):
							if question["answers"][j]:
								pass
							else:
								question["answers"][j] = [filler]
							batchAnsVec.append(question["answers"][j])
							AnsOption.append(0)
						AnsOption[question["correct_index"]] = 1
						batchAnsOpt.append(AnsOption)
						cost = 0

						if len(batchP) == batchSize:
								batchcount_epoch+=1
								global_step+=1

								batchP = varSentencePadding(batchP,max_len[0])
								batchQ = batchPadding(batchQ,max_len[1])
								batchAnsVec =  batchPadding(batchAnsVec,max_len[2])

								cost = ACMNet_object.train(batchP,batchQ,batchAnsVec,batchAnsOpt,dropoutRate)

								totalcost_epoch+=cost
								print ('global_step '+str(global_step)+' ,Cost of Epoch ' + str(i) + ' batch ' + str(batchcount_epoch) + ": "+str(cost))

								batchP = []
								batchQ = []
								batchAnsVec = []
								batchAnsOpt = []
								if global_step%200 == 0:
									accuracy_list['val'].append(test(ACMNet_object,plotFilePath,val_q,batchSize,max_len,choice,True))
									print('Val Accuracy: ',accuracy_list['val'])
									if np.argmax(np.asarray(accuracy_list['val'])) == len(accuracy_list['val'])-1:
										ACMNet_object.save(global_step)
									if i > 30 and global_step%400 == 0:
										accuracy_list['train'].append(test(ACMNet_object,plotFilePath,train_q,batchSize,max_len,choice,False))
										print('Train Accuracy: ',accuracy_list['train'])                                   

		return accuracy_list


						


