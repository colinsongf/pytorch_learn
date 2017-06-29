# coding=utf-8
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pylab
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

# 参数定义
LR = 0.02
EPOCH = 2
filepath = 'ChineseCorpus199801_cjpatch.txt'
READ_PARMS = False
WRITE_PARM = True
train_num = 100000      # 需要用于训练的数据量
EMBEDDING_DIM = 200      # 词典的扩展维数
HIDDEN_DIM = 200          # LSTM的隐藏层维数
TAG_MERGE = False       # 细分标注合并
USE_GPU = True



def printlist(data):
    for i, x in enumerate(data):
        print('%3d:%s$' % (i, x))


# 读取文件
file = open(filepath, 'r')
data = file.readlines()
data = [line for line in data if line.strip() != '']
training_data = data[:train_num]
training_data = [line.decode('gbk') for line in training_data]

# 以两个空格为分隔符，并去掉最后一个换行的回车符
training_data = [line.split('  ')[1:-1] for line in training_data]
# printlist(training_data[0])

# 将词和标签分开
training_data = [[word.split('/') for word in line] for line in training_data]

word_to_ix = {}
tag_to_ix = {}

# 语料库里有嵌套标注的情况，目前只取里面的标注
for line in training_data:
    for x in line:
        x[0] = x[0] if x[0] != '[' else x[0][1:]
        x[1] = x[1] if ']' not in x[1] else x[1].split(']')[0]
        # 生成词典
        if x[0] not in word_to_ix:
            word_to_ix[x[0]] = len(word_to_ix)
        # 生成标签词典
        if x[1] not in tag_to_ix:
            tag_to_ix[x[1]] = len(tag_to_ix)


# 语料库里有标注细分，这里合并细分标注
if TAG_MERGE:
    pass    # 还没实现

vocab_size = len(word_to_ix)
print('词典数：%5d' % vocab_size)

tag_size = len(tag_to_ix)
print('标签数：%5d' % tag_size)

# printlist(tag_to_ix)
# exit()



# 使用词典转换原始数据的函数
def prepare_sequence(line):
    _word_seq = [word_to_ix[x[0]] for x in line]
    _tag_seq = [tag_to_ix[x[1]] for x in line]

    if USE_GPU:
        _tensor1 = torch.cuda.LongTensor(_word_seq)
        _tensor2 = torch.cuda.LongTensor(_tag_seq)
    else:
        _tensor1 = torch.LongTensor(_word_seq)
        _tensor2 = torch.LongTensor(_tag_seq)
    return autograd.Variable(_tensor1), autograd.Variable(_tensor2)


# LSTM网络定义
class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        # Embedding： A simple lookup table that stores embeddings of a fixed dictionary and size.
        # class torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None,
        # max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False)
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # LSTM： Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # 第一个是h_0 ，第二个是c_0
        """
        h_0 (num_layers * num_directions, batch, hidden_size):
                tensor containing the initial hidden state for each element in the batch.
        c_0 (num_layers * num_directions, batch, hidden_size):
                tensor containing the initial cell state for each element in the batch.
        """
        if USE_GPU:
            return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim).cuda()),
                    autograd.Variable(torch.zeros(1, 1, self.hidden_dim).cuda()))
        else:
            return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                    autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden
        )
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        # log_softmax: f_i(x) = log(exp(x_i)/a), where a = sum_j exp(x_j)
        tag_scores = F.log_softmax(tag_space)
        return tag_scores


model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, vocab_size, tag_size)
if USE_GPU:
    model = model.cuda()
if READ_PARMS:
    model.load_state_dict(torch.load('lstm_params.pkl'))

# 定义损失函数 loss(x, class) = -x[class]
loss_function = nn.NLLLoss()

# 定义优化器 SGD：stochastic gradient descent
# class torch.optim.SGD(params, lr=<object object>, momentum=0, dampening=0, weight_decay=0, nesterov=False)
optimizer = optim.SGD(model.parameters(), lr=LR)

# 计算测试
word_seq, tag_seq = prepare_sequence(training_data[0])
tag_scores = model(word_seq)
loss = loss_function(tag_scores, tag_seq)
print(loss)


for line in training_data:
    word_seq, tag_seq = prepare_sequence(line)

SHOW_TIMES = 200

for epoch in range(EPOCH):

    running_loss = 0.0
    for i, line in enumerate(training_data):
        word_seq, tag_seq = prepare_sequence(line)

        model.zero_grad()
        model.hidden = model.init_hidden()

        tag_scores = model(word_seq)

        loss = loss_function(tag_scores, tag_seq)
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]
        if i % SHOW_TIMES == SHOW_TIMES-1:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / SHOW_TIMES))
            running_loss = 0.0

if WRITE_PARM:
    torch.save(model.state_dict(), 'lstm_params.pkl')

word_seq, tag_seq = prepare_sequence(training_data[0])
tag_scores = model(word_seq)
loss = loss_function(tag_scores, tag_seq)
print(loss)
