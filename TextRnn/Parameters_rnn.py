#encoding:utf-8
class parameters(object):
    """RNN配置参数"""

    # 模型参数
    embedding_dim = 100      # 词向量维度 100
    #embedding_dim = 128      # 词向量维度
    #embedding_dim = 64      
    num_classes = 2        # 类别数
    vocab_size = 10000       # 词汇表达小
    pre_trianing = None      #use vector_char trained by word2vec
    seq_length = 250
    num_layers= 2          # 隐藏层层数 best 2
    #hidden_dim = 100        # 隐藏层神经元
    hidden_dim = 100

    keep_prob = 0.5       # dropout保留比例  这个要调整，和decay 防止过拟合   原始是0.5  最好的是0.5
    #keep_prob = 0.8        
    #learning_rate = 0.001    # 学习率 1:1 1:3
    #learning_rate = 0.01  1:5
    learning_rate = 0.001  #0.001、、、0.0005的效果最好
    clip = 5           #修剪梯度 5
    #lr_decay = 0.9           #learning rate decay  学习率衰减因子  [0.01 0.1 0.5] 1:1，1:3   
    #lr_decay = 0.9
    #lr_decay = 0.7   1:5
    lr_decay = 0.7    #best 0.01  0.7
    batch_size = 16         # 每批训练大小 64
    #num_epochs = 3           # 总迭代轮次
  #  num_epochs = 15       1:5  1:1 15  1:3  25
    num_epochs = 5     ##多一点
    '''


    train_filename='E:/Easy_TextCnn_Rnn-master1/data/cnews.train.txt'  #train data
    test_filename='E:/Easy_TextCnn_Rnn-master1/data/cnews.test.txt'    #test data
    val_filename='E:/Easy_TextCnn_Rnn-master1/data/cnews.val.txt'      #validation data
    vocab_filename='E:/Easy_TextCnn_Rnn-master1/data/vocab_word.txt'        #vocabulary
    vector_word_filename='E:/Easy_TextCnn_Rnn-master1/data/vector_word.txt'  #vector_word trained by word2vec
    vector_word_npz='E:/Easy_TextCnn_Rnn-master1/data/vector_word.npz'   # save vector_word to numpy file
    '''
    
    
    #train_filename='E:/Easy_TextCnn_Rnn-master1/data/train-datasets5.tsv'  #train data
    train_filename='E:/Easy_TextCnn_Rnn-master1/data-new/new-dataset.tsv'
    #test_filename='E:/Easy_TextCnn_Rnn-master1/data/test.tsv'    #test data
    test_filename='E:/Easy_TextCnn_Rnn-master1/data-new/test.tsv'    #test data
    #val_filename='E:/Easy_TextCnn_Rnn-master1/data/val.tsv'      #validation data 
    val_filename='E:/Easy_TextCnn_Rnn-master1/data-new/val-19.tsv'      #validation data
    '''
    train_filename='E:/Easy_TextCnn_Rnn-master1/data/cnews.train.txt'  #train data
    test_filename='E:/Easy_TextCnn_Rnn-master1/data/cnews.test.txt'    #test data
    val_filename='E:/Easy_TextCnn_Rnn-master1/data/cnews.val.txt'      #validation data
    '''
    vocab_filename='E:/Easy_TextCnn_Rnn-master1/data-new/vocab_word-new4.txt'        #vocabulary
    #vocab_filename='E:/Easy_TextCnn_Rnn-master1/vocab_word.txt'        #vocabulary
    vector_word_filename='E:/Easy_TextCnn_Rnn-master1/data-new/corpusSegDone.vector'  #vector_word trained by word2vec
    vector_word_npz='E:/Easy_TextCnn_Rnn-master1/data-new/vector_word-new.npz'   # save vector_word to numpy file
    '''
    vocab_filename='E:/Easy_TextCnn_Rnn-master1/data/vocab_word.txt'        #vocabulary
    vector_word_filename='E:/Easy_TextCnn_Rnn-master1/data/vector_word.txt'  #vector_word trained by word2vec
    vector_word_npz='E:/Easy_TextCnn_Rnn-master1/data/vector_word.npz'   # save vector_word to numpy file
    '''
    
