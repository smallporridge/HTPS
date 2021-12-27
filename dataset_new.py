import torch
import pickle
import numpy as np
import os
from Models import Encoder, Contextual
import Constants
import torch
import torch.nn as nn
from metric import AP
in_path = 'data_path_folder'
filenames = sorted(os.listdir(in_path))
vocab = pickle.load(open('vocab.dict', 'rb'))
torch.manual_seed(666) # cpu
torch.cuda.manual_seed(666)
cudaid=1
#device_ids = [0, 1]
batch_size = 100
max_querylen = 20
max_doclen = 30
max_qdlen = 50
max_hislen = 50
max_sessionlen = 20
def collate_fn_train(insts):
	''' Pad the instance to the max seq length in batch '''
	querys, docs1, docs2, label, next_q, features1, features2, delta, long_qdids, longpos, short_qdids, shortpos = zip(*insts)

	querys = torch.LongTensor(querys)
	docs1 = torch.LongTensor(docs1)
	docs2 = torch.LongTensor(docs2)
	label = torch.LongTensor(label)
	next_q = torch.LongTensor(next_q)
	features1 = torch.FloatTensor(features1)
	features2 = torch.FloatTensor(features2)
	delta = torch.FloatTensor(delta)
	long_qdids = torch.LongTensor(long_qdids)
	longpos = torch.LongTensor(longpos)
	short_qdids = torch.LongTensor(short_qdids)
	shortpos = torch.LongTensor(shortpos)


	return querys, docs1, docs2, label, next_q, features1, features2, delta, long_qdids, longpos, short_qdids, shortpos

class Dataset_train(torch.utils.data.Dataset):
	def __init__(
		self, querys, docs1, docs2, label, next_q, features1, features2, delta, long_qdids, longpos, short_qdids, shortpos):
		self.querys = querys
		self.docs1 = docs1
		self.docs2 = docs2
		self.label = label
		self.next_q = next_q
		self.features1 = features1
		self.features2 = features2
		self.delta = delta
		self.long_qdids = long_qdids
		self.longpos = longpos
		self.short_qdids = short_qdids
		self.shortpos = shortpos

	def __len__(self):
		return len(self.querys)

	def __getitem__(self, idx):
		querys = self.querys[idx]
		docs1 = self.docs1[idx]
		docs2 = self.docs2[idx]
		label = self.label[idx]
		next_q = self.next_q[idx]
		features1 = self.features1[idx]
		features2 = self.features2[idx]
		delta = self.delta[idx]
		long_qdids = self.long_qdids[idx]
		longpos = self.longpos[idx]
		short_qdids = self.short_qdids[idx]
		shortpos = self.shortpos[idx]
		return querys, docs1, docs2, label, next_q, features1, features2, delta, long_qdids, longpos, short_qdids, shortpos

def collate_fn_score(insts):
	''' Pad the instance to the max seq length in batch '''
	querys, docs, features, long_qdids, longpos, short_qdids, shortpos, lines = zip(*insts)

	querys = torch.LongTensor(querys)
	docs = torch.LongTensor(docs)
	features = torch.FloatTensor(features)
	long_qdids = torch.LongTensor(long_qdids)
	longpos = torch.LongTensor(longpos)
	short_qdids = torch.LongTensor(short_qdids)
	shortpos = torch.LongTensor(shortpos)

	return querys, docs, features, long_qdids, longpos, short_qdids, shortpos, lines

class Dataset_score(torch.utils.data.Dataset):
	def __init__(
		self, querys, docs, features, long_qdids, longpos, short_qdids, shortpos, lines):
		self.querys = querys
		self.docs = docs
		self.features = features
		self.lines = lines
		self.long_qdids = long_qdids
		self.longpos = longpos
		self.short_qdids = short_qdids
		self.shortpos = shortpos

	def __len__(self):
		return len(self.querys)

	def __getitem__(self, idx):
		querys = self.querys[idx]
		docs = self.docs[idx]
		features = self.features[idx]
		lines = self.lines[idx]
		long_qdids = self.long_qdids[idx]
		longpos = self.longpos[idx]
		short_qdids = self.short_qdids[idx]
		shortpos = self.shortpos[idx]
		return querys, docs, features, long_qdids, longpos, short_qdids, shortpos, lines

def sen2qid(sen):
	idx = []
	for word in sen.split():
		if word in vocab:
			idx.append(vocab[word])
		else:
			idx.append(vocab['<unk>'])
	idx = idx[:max_querylen]
	padding = [0] * (max_querylen - len(idx))
	idx = idx + padding
	return	idx

def sen2did(sen):
	idx = []
	for word in sen.split():
		if word in vocab:
			idx.append(vocab[word])
		else:
			idx.append(vocab['<unk>'])
	idx = idx[:max_doclen]
	padding = [0] * (max_doclen - len(idx))
	idx = idx + padding
	return	idx

def sen2id(sen):
	idx = []
	for word in sen.split():
		if word in vocab:
			idx.append(vocab[word])
		else:
			idx.append(vocab['<unk>'])
	idx = idx[:max_qdlen]
	padding = [0] * (max_qdlen - len(idx))
	idx = idx + padding
	return	idx

def divide_dataset(filename):
	session_sum = 0
	query_sum = 0 
	last_queryid = 0
	last_sessionid = 0
	with open(os.path.join(in_path, filename)) as fhand:
		for line in fhand:
			try:
				line, features = line.strip().split('###')
			except:
				line = line.strip()
			user, sessionid, querytime, query, url, title, sat, urlrank = line.strip().split('\t') 
			queryid = sessionid + querytime + query

			if querytime < '2006-04-03 00:00:00':
				if queryid != last_queryid:
					last_queryid = queryid
				query_sum += 1
			elif querytime < '2006-05-16 00:00:00':
				if query_sum < 2:
					return False
				if sessionid != last_sessionid:
					session_sum += 1
					assert(last_queryid != queryid)
					last_sessionid = sessionid
				if queryid != last_queryid:
					last_queryid = queryid
			else:
				if sessionid != last_sessionid:  # 这里不区分valid 和 test
					session_sum += 1
					assert(last_queryid != queryid)
					last_sessionid = sessionid
				if queryid != last_queryid:
					last_queryid = queryid

		if session_sum < 4:
			return False
	return True

q_train = []
querys_train = []
docs1_train = []
docs2_train = []
title1_train = []
title2_train = []
label_train = []
delta_train = []
features1_train = []
features2_train = []
long_qdids_train = []
longpos_train = []
short_qdids_train = []
shortpos_train = []

querys_test = []
docs_test = []
features_test = []
lines = []
long_qdids_test = []
longpos_test = []
short_qdids_test = []
shortpos_test = []

def prepare_pairdata(sat_list, feature_list, doc_list, qids, long_qdids, short_qdids):
	short_qdids_his = short_qdids[-max_sessionlen:]
	long_qdids_his = long_qdids[-max_hislen:]

	init_short_qdids = np.zeros((max_sessionlen,max_qdlen))
	init_long_qdids = np.zeros((max_hislen,max_qdlen))

	init_shortpos = np.zeros(max_sessionlen+1)
	init_shortpos[-1] = max_sessionlen+1
	init_longpos = np.zeros(max_hislen+1)
	init_longpos[-1] = max_hislen+1

	for i in range(max_sessionlen):
		init_short_qdids[i][0]=1
	for i in range(len(short_qdids_his)):
		init_short_qdids[i] = short_qdids_his[i]
		init_shortpos[i] = i+1
	for i in range(max_hislen):
		init_long_qdids[i][0]=1
	for i in range(len(long_qdids_his)):
		init_long_qdids[i] = long_qdids_his[i]
		init_longpos[i] = i+1
		
	delta = cal_delta(sat_list)
	n_targets = len(sat_list)
	for i in range(n_targets):
		for j in range(i+1, n_targets):
			if delta[i, j]>0:
				rel_doc = doc_list[j]
				rel_features = feature_list[j]
				irr_doc = doc_list[i]
				irr_features = feature_list[i]
				lbd = delta[i, j]
			elif delta[i, j]<0:
				rel_doc = doc_list[i]
				rel_features = feature_list[i]
				irr_doc = doc_list[j]
				irr_features = feature_list[j]
				lbd = -delta[i, j]
			else:
				continue
			if True:
				short_qdids_train.append(init_short_qdids)
				shortpos_train.append(init_shortpos)
				long_qdids_train.append(init_long_qdids)
				longpos_train.append(init_longpos)
				querys_train.append(qids)
				docs1_train.append(rel_doc)
				docs2_train.append(irr_doc)
				label_train.append(0)
				features1_train.append(rel_features)
				features2_train.append(irr_features)
				delta_train.append(lbd)

def predata():
	x_train = []
	filenum=0
	key=0
	for filename in filenames:
		if not divide_dataset(filename):
			continue
		filenum += 1
		if filenum % 100 == 0:
			print(filenum)
		last_queryid = 0
		last_sessionid = 0
		last_qids = 0
		queryid = 0
		sessionid = 0
		key = 0
		satcount = 0
		intent = ''
		doc_list = []
		sat_list = []
		feature_list = []
		long_qdids = []
		short_qdids = []

		fhand = open(os.path.join(in_path, filename))
		for line in fhand:
			try:
				line, features = line.strip().split('###')
				features = [float(item) for item in features.split('\t')]
				features = features[:14]+features[26:]
				if np.isnan(np.array(features)).sum():
					continue
			except:
				line = line.strip()
				features = [0]*98
			user, sessionid, querytime, query, url, title, sat, urlrank = line.strip().split('\t')
			queryid = sessionid + querytime + query
			qids = sen2qid(query)
			dids = sen2did(title)
			if querytime <= '2006-05-16 00:00:00':
				if queryid != last_queryid:
					if key == 1 and querytime >= '2006-04-03 00:00:00': #There is a SAT-click in the sesssion
						prepare_pairdata(sat_list, feature_list, doc_list, last_qids, long_qdids, short_qdids)
						key = 0
					if intent != '':
						qdids = sen2id(intent)
						short_qdids.append(qdids)
					intent = query
					doc_list = []
					sat_list = []
					feature_list = []
					last_queryid = queryid
					last_qids = qids

					if sessionid != last_sessionid:
						if len(short_qdids) != 0:
							long_qdids.extend(short_qdids)
						short_qdids = []
						last_sessionid = sessionid
				doc_list.append(dids)
				sat_list.append(sat)
				feature_list.append(features)
				if int(sat) == 1:
					intent += ' ' + title
					key = 1
			else:
				if queryid != last_queryid:
					if intent != '':
						qdids = sen2id(intent)
						short_qdids.append(qdids)
					last_queryid = queryid
					intent = query
					if sessionid != last_sessionid:
						if len(short_qdids) != 0:
							long_qdids.extend(short_qdids)
						short_qdids = []
						last_sessionid = sessionid
				if int(sat) == 1:
					intent += ' ' + title
				short_qdids_his = short_qdids[-max_sessionlen:]
				long_qdids_his = long_qdids[-max_hislen:]
				init_short_qdids = np.zeros((max_sessionlen,max_qdlen))
				init_long_qdids = np.zeros((max_hislen,max_qdlen))
				init_shortpos = np.zeros(max_sessionlen+1)
				init_shortpos[-1] = max_sessionlen+1
				init_longpos = np.zeros(max_hislen+1)
				init_longpos[-1] = max_hislen+1
				init_sessionpos = np.zeros(max_sessionlen+1)
				init_sessionpos[-1] = max_sessionlen+1
				for i in range(max_sessionlen):
					init_short_qdids[i][0]=1
				for i in range(len(short_qdids_his)):
					init_short_qdids[i] = short_qdids_his[i]
					init_shortpos[i] = i+1
				for i in range(max_hislen):
					init_long_qdids[i][0]=1
				for i in range(len(long_qdids_his)):
					init_long_qdids[i] = long_qdids_his[i]
					init_longpos[i] = i+1

				short_qdids_test.append(init_short_qdids)
				shortpos_test.append(init_shortpos)
				long_qdids_test.append(init_long_qdids)
				longpos_test.append(init_longpos)
				querys_test.append(qids)
				docs_test.append(dids)
				features_test.append(features)
				lines.append(line.strip('\n'))

model = Contextual(max_querylen=max_querylen, max_qdlen=max_qdlen, max_hislen=max_hislen, max_sessionlen=max_sessionlen, batch_size=batch_size, d_word_vec=100,
				n_layers=1, n_head=6, d_k=50, d_v=50,
				d_model=100, d_inner=256, dropout=0.1)
#model = torch.nn.DataParallel(model, device_ids=device_ids)
model = model.cuda(cudaid)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
def train(train_loader):
	model.train()
	for batch_idx, (query, docs1, docs2, label, next_q, features1, features2, delta, long_qdids, longpos, short_qdids, shortpos) in enumerate(train_loader):
		optimizer.zero_grad()

		query = query.cuda(cudaid)
		docs1 =docs1.cuda(cudaid)
		docs2 = docs2.cuda(cudaid)
		label = label.cuda(cudaid)
		next_q = next_q.cuda(cudaid)
		features1 = features1.cuda(cudaid)
		features2 = features2.cuda(cudaid)
		long_qdids = long_qdids.cuda(cudaid)
		longpos = longpos.cuda(cudaid)
		short_qdids = short_qdids.cuda(cudaid)
		shortpos = shortpos.cuda(cudaid)

		score, pre, p_score = model(query, docs1, docs2, features1, features2, long_qdids, longpos, short_qdids, shortpos)
		correct = torch.max(pre,1)[1].eq(label).cpu().sum()
		loss = criterion(p_score, label)# + loss_2/2.0
		loss.backward()
		optimizer.step()
		print(loss)
	torch.save(model.state_dict(),'model.zyj')

def score(test_loader, load=False):
	if load == True:
		model.load_state_dict(torch.load('model.zyj'))
	model.eval()
	test_loss = 0
	correct = 0
	f = open('test_score.txt','w')
	for query, docs, features, long_qdids, longpos, short_qdids, shortpos, lines in test_loader:

		query = query.cuda(cudaid)
		docs =docs.cuda(cudaid)
		features = features.cuda(cudaid)
		long_qdids = long_qdids.cuda(cudaid)
		longpos = longpos.cuda(cudaid)
		short_qdids = short_qdids.cuda(cudaid)
		shortpos = shortpos.cuda(cudaid)

		score, pre, p_score = model(query, docs, docs, features, features, long_qdids, longpos, short_qdids, shortpos)
		for line, sc in zip(lines, score):
			f.write(line+'\t'+str(float(sc[0]))+'\n')


if __name__ == '__main__':
	predata()
	train_loader = torch.utils.data.DataLoader(
		Dataset_train(
			querys=querys_train, docs1=docs1_train, docs2=docs2_train, label=label_train, next_q=next_q_train,
			features1=features1_train, features2=features2_train, delta=delta_train, long_qdids=long_qdids_train, longpos=longpos_train,
			short_qdids=short_qdids_train, shortpos=shortpos_train),
		batch_size=batch_size,
		collate_fn=collate_fn_train)

	score_loader = torch.utils.data.DataLoader(
		Dataset_score(
			querys=querys_test, docs=docs_test, features=features_test, long_qdids=long_qdids_test, longpos=longpos_test,
			short_qdids=short_qdids_test, shortpos=shortpos_test, lines=lines),
		batch_size=64,
		collate_fn=collate_fn_score)
	evaluation = AP()
	# model.load_state_dict(torch.load('model.zyj'))
	# with open('test_score.txt', 'r') as f:
	# 	evaluation.evaluate(f)
	#score(score_loader)
	for epoch in range(1):
		train(train_loader)
		score(score_loader)
		with open('test_score.txt', 'r') as f:
			evaluation.evaluate(f)

	
	
