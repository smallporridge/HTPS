''' Define the Transformer model '''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import Constants
import pickle
from pytorch_pretrained_bert import BertModel, BertTokenizer
from Layers import EncoderLayer, DecoderLayer
from torch.autograd import Variable

__author__ = "Yu-Hsiang Huang"
cudaid=1
vocabulary = pickle.load(open('/home/yujia_zhou/pytorch/transformer/data/vocab.dict', 'rb'))

def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask

def kernel_mus(n_kernels):
        l_mu = [1]
        if n_kernels == 1:
            return l_mu
        bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
        l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
        for i in range(1, n_kernels - 1):
            l_mu.append(l_mu[i] - bin_size)
        return l_mu

def kernel_sigmas(n_kernels):
    bin_size = 2.0 / (n_kernels - 1)
    l_sigma = [0.001]  # for exact match. small variance -> exact match
    if n_kernels == 1:
        return l_sigma
    l_sigma += [0.1] * (n_kernels - 1)
    return l_sigma

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            n_src_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()

        n_position = len_max_seq + 1

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)

        # -- Forward
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,

class knrm(nn.Module):
    def __init__(self, k):
        super(knrm, self).__init__()
        tensor_mu = torch.FloatTensor(kernel_mus(k)).cuda(cudaid)
        tensor_sigma = torch.FloatTensor(kernel_sigmas(k)).cuda(cudaid)
        self.mu = Variable(tensor_mu, requires_grad = False).view(1, 1, 1, k)
        self.sigma = Variable(tensor_sigma, requires_grad = False).view(1, 1, 1, k)
        self.dense = nn.Linear(k, 1, 1)

    def get_intersect_matrix(self, q_embed, d_embed, attn_q, attn_d):
        sim = torch.bmm(q_embed, torch.transpose(d_embed, 1, 2)).view(q_embed.size()[0], q_embed.size()[1], d_embed.size()[1], 1) # n*m*d*1
        pooling_value = torch.exp((- ((sim - self.mu) ** 2) / (self.sigma ** 2) / 2)) * attn_d
        pooling_sum = torch.sum(pooling_value, 2)
        log_pooling_sum = torch.log(torch.clamp(pooling_sum, min=1e-10)) * 0.01 * attn_q
        log_pooling_sum = torch.sum(log_pooling_sum, 1)#soft-TF
        return log_pooling_sum

    def forward(self, inputs_q, inputs_d, mask_q, mask_d):
        q_embed_norm = F.normalize(inputs_q, 2, 2)
        d_embed_norm = F.normalize(inputs_d, 2, 2)
        mask_d = mask_d.view(mask_d.size()[0], 1, mask_d.size()[1], 1)
        mask_q = mask_q.view(mask_q.size()[0], mask_q.size()[1], 1)
        log_pooling_sum = self.get_intersect_matrix(q_embed_norm, d_embed_norm, mask_q, mask_d)
        output = F.tanh(self.dense(log_pooling_sum))
        return output

class Encoder_high(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()

        n_position = len_max_seq + 1

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_emb, src_pos, return_attns=False, needpos=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_pos, seq_q=src_pos)
        non_pad_mask = get_non_pad_mask(src_pos)

        # -- Forward
        if needpos:
            enc_output = src_emb + self.position_enc(src_pos)
        else:
            enc_output = src_emb

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,

class Contextual(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def load_embedding(self): # load the pretrained embedding
        weight = torch.zeros(len(vocabulary)+1, self.d_word_vec)
        weight[-1] = torch.rand(self.d_word_vec)
        with open('/home/yujia_zhou/pytorch/transformer/data/word2vec.txt', 'r') as fr:
            for line in fr:
                line = line.strip().split()
                wordid = vocabulary[line[0]]
                weight[wordid, :] = torch.FloatTensor([float(t) for t in line[1:]]) 
        print("Successfully load the word vectors...")
        return weight

    def __init__(
            self, max_querylen, max_qdlen, max_hislen, max_sessionlen, batch_size,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1):

        super().__init__()

        self.max_qdlen = max_qdlen
        self.max_querylen = max_querylen
        self.max_hislen = max_hislen
        self.max_sessionlen = max_sessionlen
        self.d_word_vec = d_word_vec

        self.knrm_1 = knrm(11)
        self.knrm_2 = knrm(11)

        self.encoder_query = Encoder_high(
            len_max_seq=max_qdlen,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.encoder_short_his = Encoder_high(
            len_max_seq=max_sessionlen+1,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.encoder_long_his = Encoder_high(
            len_max_seq=max_hislen+1,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.encoder_short_query = Encoder_high(
            len_max_seq=max_sessionlen+1,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.encoder_long_query = Encoder_high(
            len_max_seq=max_hislen+max_sessionlen+1,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        #self.linear_layer=nn.Sequential(nn.Linear(d_word_vec, 2, bias = False),nn.Tanh())
        self.feature_layer=nn.Sequential(nn.Linear(98, 1),nn.Tanh())
        self.gate=nn.Sequential(nn.Linear(self.d_word_vec*2, 1),nn.Sigmoid())
        self.score_layer=nn.Linear(4, 1)
        #self.embedding = nn.Embedding.from_pretrained(self.load_embedding())
        self.embedding = nn.Embedding(len(vocabulary)+1, self.d_word_vec)
        self.embedding.weight.data.copy_(self.load_embedding())

    def pairwise_loss(self, score1, score2):
        return (1/(1+torch.exp(score2-score1)))


    def forward(self, query, docs1, docs2, features1, features2, long_qdids, longpos, short_qdids, shortpos):
        '''
        query: query word ids
        docs1: positive document word ids
        docs2: negative document word ids
        features1: positive document additional features
        features2: negative document additional features
        long_qdis: query and document ids in long-term history
        longpos: position id
        short_qdis: query and document ids in short-term history 
        shortpos: position id
        '''

        all_qdids = torch.cat([long_qdids, short_qdids], 1)
        all_qd_mask = all_qdids.view(-1,self.max_qdlen)

        qenc_output_1 = self.embedding(query)
        d1enc_output_1 = self.embedding(docs1)
        d2enc_output_1 = self.embedding(docs2)
        all_qdenc = self.embedding(all_qdids)

        all_qdenc = all_qdenc.view(-1, self.max_qdlen, self.d_word_vec)
        all_qdenc, *_ = self.encoder_query(all_qdenc, all_qd_mask)
        qenc_output_2, *_ = self.encoder_query(qenc_output_1, query)
        d1enc_output_2, *_ = self.encoder_query(d1enc_output_1, docs1)
        d2enc_output_2, *_ = self.encoder_query(d2enc_output_1, docs2)

        d1enc_output_3 = torch.sum(d1enc_output_2, 1)
        d2enc_output_3 = torch.sum(d2enc_output_2, 1)
        qenc_output_3 = torch.sum(qenc_output_2, 1)
        all_qdenc = torch.sum(all_qdenc, 1)
        long_qdenc, short_qdenc = torch.split(all_qdenc, [self.max_hislen, self.max_sessionlen], 1)


        sq_qdenc = torch.cat([short_qdenc, qenc_output_3.unsqueeze(1)], 1)
        sq_qdenc, *_ = self.encoder_short_his(sq_qdenc, shortpos, needpos=True)
        short_qdenc, qenc_output_4 = torch.split(sq_qdenc, [self.max_sessionlen, 1], 1)

        lq_qdenc = torch.cat([long_qdenc, qenc_output_4], 1)
        lq_qdenc, *_ = self.encoder_long_his(lq_qdenc, longpos, needpos=True)
        long_qdenc, qenc_output_5 = torch.split(sq_qdenc, [self.max_sessionlen, 1], 1)


        q_mask = get_non_pad_mask(query)
        d1_mask = get_non_pad_mask(docs1)
        d2_mask = get_non_pad_mask(docs2)

        score_1_1 = self.knrm_1(qenc_output_1, d1enc_output_1, q_mask, d1_mask)
        score_2_1 = self.knrm_1(qenc_output_1, d2enc_output_1, q_mask, d2_mask)

        score_1_2 = self.knrm_2(qenc_output_2, d1enc_output_2, q_mask, d1_mask)
        score_2_2 = self.knrm_2(qenc_output_2, d2enc_output_2, q_mask, d2_mask)

        score_1_3 = torch.cosine_similarity(qenc_output_3, d1enc_output_3, dim=1).unsqueeze(1)
        score_2_3 = torch.cosine_similarity(qenc_output_3, d2enc_output_3, dim=1).unsqueeze(1)

        score_1_4 = torch.cosine_similarity(qenc_output_4.squeeze(1), d1enc_output_3, dim=1).unsqueeze(1)
        score_2_4 = torch.cosine_similarity(qenc_output_4.squeeze(1), d2enc_output_3, dim=1).unsqueeze(1)

        score_1_5 = torch.cosine_similarity(qenc_output_5.squeeze(1), d1enc_output_3, dim=1).unsqueeze(1)
        score_2_5 = torch.cosine_similarity(qenc_output_5.squeeze(1), d2enc_output_3, dim=1).unsqueeze(1)

        score_1_8 = self.feature_layer(features1)
        score_2_8 = self.feature_layer(features2)

        score1_all = torch.cat([score_1_1, score_1_2, score_1_3, score_1_8], 1)
        score2_all = torch.cat([score_2_1, score_2_2, score_2_3, score_2_8], 1)
        score_1 = self.score_layer(score1_all)
        score_2 = self.score_layer(score2_all)

        score = torch.cat([score_1, score_2], 1)

        p_score = torch.cat([self.pairwise_loss(score_1, score_2),
                    self.pairwise_loss(score_2, score_1)], 1)

        pre = F.softmax(score, 1)

        return score, pre, p_score


