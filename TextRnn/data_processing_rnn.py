#encoding:utf-8
from collections import Counter
import tensorflow.contrib.keras as kr
import numpy as np
import codecs
import re
import jieba


def read_file(filename):
    re_han = re.compile(u"([\u4E00-\u9FD5a-zA-Z0-9+#&\._%]+)")  # the method of cutting text by punctuation
    contents, labels = [], []
    with codecs.open(filename, 'r', encoding='utf-8-sig') as f:
        for line in f:
            try:
                line = line.rstrip()
                assert len(line.split('\t')) == 2
                label, content = line.split('\t')
                labels.append(label)
                blocks = re_han.split(content)
                word = []
                for blk in blocks:
                    if re_han.match(blk):
                        word.extend(jieba.lcut(blk))
                contents.append(word)
            except:
                pass
    return labels, contents

#def built_vocab_vector(filenames,voc_size = 10000):
def built_vocab_vector(filenames,voc_size = 5001):
    '''
    去停用词，得到前9999个词，获取对应的词 以及 词向量
    :param filenames:
    :param voc_size:
    :return:
    '''
    #stopword = open('E:/Easy_TextCnn_Rnn-master1/data/stopwords.txt', 'r', encoding='utf-8-sig')
    stopword = open(r'E:\积水微博\负样本文件夹\测试0\分词测试\停用词新.txt','r',encoding='gb18030',errors='ignore')
    stop = [key.strip(' \n') for key in stopword]

    all_data = []
    j = 1
    #embeddings = np.zeros([10000, 100])
    embeddings = np.zeros([5001, 100])

    for filename in filenames:
        labels, content = read_file(filename)
        for eachline in content:
            line =[]
            for i in range(len(eachline)):
                if str(eachline[i]) not in stop:#去停用词
                    line.append(eachline[i])
            all_data.extend(line)

    counter = Counter(all_data)
    count_paris = counter.most_common(voc_size-1)
    word, _ = list(zip(*count_paris))

    f = codecs.open('E:/Easy_TextCnn_Rnn-master1/data-new/corpusSegDone.vector.txt', 'r', encoding='utf-8-sig')
    vocab_word = open('E:/Easy_TextCnn_Rnn-master1/data-new/vocab_word-new.txt', 'w', encoding='utf-8-sig')
    for ealine in f:
        item = ealine.split(' ')
        key = item[0]
        vec = np.array(item[1:], dtype='float32')
        if key in word:
            embeddings[j] = np.array(vec)
            vocab_word.write(key.strip('\r') + '\n')
            j += 1
    np.savez_compressed('E:/Easy_TextCnn_Rnn-master1/data-new/vector_word-new.npz', embeddings=embeddings)

    

    

def get_wordid(filename):
    key = open(filename, 'r', encoding='utf-8-sig')

    wordid = {}
    wordid['<PAD>'] = 0
    j = 1
    for w in key:
        w = w.strip('\n')
        w = w.strip('\r')
        wordid[w] = j
        j += 1
    return wordid



def read_category():
    
    categories = ['正','负']

    #categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    cat_to_id = dict(zip(categories, range(len(categories))))
    return categories, cat_to_id

def process(filename, word_to_id, cat_to_id, max_length=250):
    labels, contents = read_file(filename)
    data_id, label_id = [], []

    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length, padding='post', truncating='post')
    y_pad = kr.utils.to_categorical(label_id)

    return x_pad, y_pad
'''
def process(filename, word_to_id, cat_to_id, max_length=600):    
    #contents,labels, = read_file(filename)
    labels, contents = read_file(filename)
    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length, padding='post', truncating='post')
    y_pad = kr.utils.to_categorical(label_id)
    return x_pad, y_pad
'''

def get_word2vec(filename):
    with np.load(filename) as data:
        return data['embeddings']


def batch_iter(x, y,  batch_size = 64):
    data_len = len(x)
    num_batch = int((data_len - 1)/batch_size) + 1
    indices = np.random.permutation(np.arange(data_len))
    '''
    np.arange(4) = [0,1,2,3]
    np.random.permutation([1, 4, 9, 12, 15]) = [15,  1,  9,  4, 12]
    '''
    x_shuff = x[indices]
    y_shuff = y[indices]
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i+1) * batch_size, data_len)
        yield x_shuff[start_id:end_id], y_shuff[start_id:end_id]

def sequence(x_batch):
    seq_len = []
    for line in x_batch:
        length = np.sum(np.sign(line))
        seq_len.append(length)

    return seq_len


