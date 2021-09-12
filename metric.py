import os
import pickle
import numpy as np
class Metric(object):
	"""Base metric class.
	Subclasses must override score() and can optionally override various
	other methods.
	"""
	def __init__(self):
		print("Generating the entropy dictionary.")
		with open('entropy.dict', 'rb') as fr:
			self.entropy = pickle.load(fr)
			
	def rerank(self, scores, lines):
		idxs = list(zip(*sorted(enumerate(scores), key=lambda x:x[1], reverse=True)))[0]
		final_lines = []
		for idx in idxs:
			final_lines.append(lines[idx])
		return final_lines

	def score(self, clicks):
		raise NotImplementedError()

	def evaluate(self, filehandle):
		"""
		evaluate queries in the filehadle, each line is the same as the training dataset.
		"""
		last_queryid = 0
		f = open(self.__class__.__name__+'.txt', 'w')
		re = open('results.txt','a')
		mscore1, mscore2 = 0.0, 0.0
		nquery = 0.0
		clicks = []
		scores = []
		for line in filehandle:
			user, sessionid, querytime, query, url, title, sat, _,score = line.strip().split('\t') 
			queryid = user + sessionid + querytime + query

			if queryid != last_queryid: # 表示一个query结束了
				if len(clicks) == 50:
					score1 = self.score(clicks)
					score2 = self.score(self.rerank(scores, clicks))
					if score1 != -1:
						nquery += 1
						mscore1 += score1
						mscore2 += score2
					f.write(last_queryid+'\t'+str(self.entropy[query])+'\t'+
						'\t'+str(score1)+'\t'+str(score2)+'\n')
				clicks = []
				scores = []
				last_queryid = queryid
			clicks.append(sat)
			scores.append(float(score))
		if len(clicks) != 0 and len(clicks) != 1:
			score1 = self.score(clicks)
			score2 = self.score(self.rerank(scores, clicks))
			if score1 != -1:
				nquery += 1
				mscore1 += score1
				mscore2 += score2
			f.write(last_queryid+'\t'+str(self.entropy[query])+'\t'+
				'\t'+str(score1)+'\t'+str(score2)+'\n')	
		f.close()  
		print("The "+self.__class__.__name__+" of original ranking is {}.".format(mscore1/nquery))
		print("The "+self.__class__.__name__+" of new ranking is {}.".format(mscore2/nquery))
		re.write("The "+self.__class__.__name__+" of original ranking is {}.\n".format(mscore1/nquery))
		re.write("The "+self.__class__.__name__+" of new ranking is {}.\n".format(mscore2/nquery))

	def write_score(self, scores, lines, filehandle):
		assert(len(scores[0])==len(lines))
		for i in range(len(scores[0])):
			filehandle.write(lines[i].rstrip('\n')+'\t'+str(scores[0][i][0])+'\n')

class AP(Metric):
	# 平均正确率(Average Precision)：对不同召回率点上的正确率进行平均。
	def  __init__(self, cutoff='1'):
		super(AP, self).__init__()
		self.cutoff = cutoff

	def score(self, clicks):
		num_rel = 0
		total_prec = 0.0
		for i in range(len(clicks)):
			if clicks[i] == self.cutoff or clicks[i] == int(self.cutoff):
				num_rel += 1
				total_prec += num_rel / (i + 1.0)
		return (total_prec / num_rel) if num_rel > 0 else -1

class MRR(Metric):
	# 第一个正确答案位置的倒数，取平均值
	def __init__(self, cutoff='1'):
		super(MRR, self).__init__()
		self.cutoff = cutoff

	def score(self, clicks):
		num_rel = 0
		total_prec = 0.0
		for i in range(len(clicks)):
			if clicks[i] == self.cutoff or clicks[i] == int(self.cutoff):
				num_rel = 1
				total_prec = 1.0 / (i+1)
				break
		return total_prec if num_rel > 0 else -1

class Precision(Metric):
	# precision@k: 计算前k个文档中有几个是正确文档  # 本文中这种计算方式合理嘛？
	def __init__(self, cutoff='1', k=1):
		super(Precision, self).__init__()
		self.cutoff = cutoff
		self.k = k

	def score(self, clicks):
		prec_in = 0.0
		prec_out = 0.0
		for i in range(len(clicks)):
			if clicks[i] == self.cutoff or clicks[i] == int(self.cutoff):
				if i+1 <= self.k:
					prec_in = 1
					break
				else:
					prec_out = 1
		if prec_in > 0:
			return 1
		else:
			return 0 if prec_out > 0 else -1

class AvePosition(Metric):
	# 平均位置：所有相关文档的位置的平均值
	def __init__(self, cutoff='1'):
		super(AvePosition, self).__init__()
		self.cutoff = cutoff

	def score(self, clicks):
		position = 0.0
		nclick = 0.0
		for i in range(len(clicks)):
			if clicks[i] == self.cutoff or clicks[i] == int(self.cutoff):
				position += i+1
				nclick += 1
		return (position/nclick) if nclick > 0 else -1


class Pgain(Metric):
	def __init__(self, cutoff='1'):
		super(Pgain, self).__init__()
		self.cutoff = cutoff

	def score(self, clicks, scores):
		up=0
		down=0
		pairs=0
		for i in range(len(clicks)):
			if clicks[i]==1:
				for j in range(0,i):#
					if clicks[j]==0:
						pairs+=1
					if clicks[j]==0 and scores[i]>scores[j]:
						up+=1
						
				if i+1<len(clicks):
					j=i+1
					if clicks[j]==0:
						pairs+=1
					if clicks[j]==0 and scores[i]<scores[j]:
						down+=1
		return up,down,pairs

	def evaluate(self, filehandle):
		last_queryid = 0
		#f = open(self.__class__.__name__+'.txt', 'w')
		nup = 0.0
		ndown = 0.0
		npairs = 0.0
		clicks = []
		scores = []
		for line in filehandle:
			user, sessionid, sessiontime, querynum, query_id, query, querytime, clicknum, urlrank, url, clicktime, dwell, click, score = line.rstrip().split('\t')
			if float(click)>=30:
				sat = '1'
			else:
				sat = '0'
			queryid = query_id + '_' + query + '_' + querytime
			if queryid != last_queryid:
				if len(clicks) != 0 and len(clicks)!=1:
					up,down,pairs=self.score(clicks,scores)
					nup+=up
					ndown+=down
					npairs+=pairs
				clicks = []
				scores = []
				last_queryid = queryid
			clicks.append(int(sat))
			scores.append(float(score))
		if len(clicks) != 0 and len(clicks)!=1:
			up,down,pairs=self.score(clicks,scores)
			nup+=up
			ndown+=down
			npairs+=pairs
		print("The number of better rankings is {}.".format(nup))
		print("The number of worse rankings is {}.".format(ndown))
		print("The Pgain is {}.".format((nup-ndown)/(npairs)))
