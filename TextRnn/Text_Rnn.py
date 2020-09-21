import tensorflow as tf



import sys
sys.path.append('E:/Easy_TextCnn_Rnn-master1/TextRnn')
'''python import模块时， 是在sys.path里按顺序查找的。
sys.path是一个列表，里面以字符串的形式存储了许多路径。
使用A.py文件中的函数需要先将他的文件路径放到sys.path中'''
from Parameters_rnn import parameters as pm
from data_processing_rnn import batch_iter, sequence
 
def mean_fun(onelist):
    count = 0
    for i in onelist:
        count += i
    return float(count/len(onelist))

class RnnModel(object):

    def __init__(self):
        self.input_x = tf.placeholder(tf.int32, shape=[None, pm.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, shape=[None, pm.num_classes], name='input_y')
        self.seq_length = tf.placeholder(tf.int32, shape=[None], name='sequen_length') #, shape=[None]
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.rnn()

    def rnn(self):

        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            embedding = tf.get_variable('embedding', shape=[pm.vocab_size, pm.embedding_dim],
                                        initializer=tf.constant_initializer(pm.pre_trianing))
            self.embedding_input = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope('cell'):
            cell = tf.nn.rnn_cell.LSTMCell(pm.hidden_dim)
            #cell = tf.nn.rnn_cell.BasicRNNCell(pm.hidden_dim)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

            cells = [cell for _ in range(pm.num_layers)]
            Cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)


        with tf.name_scope('rnn'):
            #hidden一层 输入是[batch_size, seq_length, hidden_dim]
            #hidden二层 输入是[batch_size, seq_length, 2*hidden_dim]
            #2*hidden_dim = embendding_dim + hidden_dim
            output, _ = tf.nn.dynamic_rnn(cell=Cell, inputs=self.embedding_input, sequence_length=self.seq_length, dtype=tf.float32)
            output = tf.reduce_sum(output, axis=1)
            #output:[batch_size, seq_length, hidden_dim]

        with tf.name_scope('dropout'):
            self.out_drop = tf.nn.dropout(output, keep_prob=self.keep_prob)

        with tf.name_scope('output'):
            w = tf.Variable(tf.truncated_normal([pm.hidden_dim, pm.num_classes], stddev=0.1), name='w')
            b = tf.Variable(tf.constant(0.1, shape=[pm.num_classes]), name='b')             
            self.logits = tf.matmul(self.out_drop, w) + b  #最后一层的输出
            
            self.probability = tf.nn.softmax(self.logits)##
            #self.probability = tf.nn.relu(self.logits)
            #self.probability = tf.nn.tanh(self.logits)
            #self.probability = tf.nn.sigmoid(self.logits)
            self.predict = tf.argmax(tf.nn.softmax(self.logits), 1, name='predict')
            #self.predict = tf.argmax(tf.nn.relu(self.logits), 1, name='predict')
            #self.predict = tf.argmax(tf.nn.tanh(self.logits), 1, name='predict')
            #self.predict = tf.argmax(tf.nn.sigmoid(self.logits), 1, name='predict')
            #self.probability = tf.argmax(tf.nn.relu(self.logits),1,name='probability')#####
        '''
        #这个是原来的代码
        with tf.name_scope('loss'):
            #losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y)  #原来的代码
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)
        '''
        #交叉熵损失函数    
        with tf.name_scope('loss'):
            #losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)  #logits表示的是神经网络最后一层的类别预测输出值
            self.loss = tf.reduce_mean(losses)  #对交叉熵取均值非常有必要            
            

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(pm.learning_rate)
        
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))#计算变量梯度，得到梯度值,变量
            gradients, _ = tf.clip_by_global_norm(gradients, pm.clip)
            #对g进行l2正则化计算，比较其与clip的值，如果l2后的值更大，让梯度*(clip/l2_g),得到新梯度
            self.optimizer = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)
            #global_step 自动+1

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(self.predict, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    def feed_data(self, x_batch, y_batch, seq_len, keep_prob):

        feed_dict = {self.input_x: x_batch,
                     self.input_y: y_batch,
                     self.seq_length: seq_len,
                     self.keep_prob: keep_prob}

        return feed_dict

    def evaluate(self, sess, x, y):
        loss_all=[]
        accuracy_all=[]

        batch_test = batch_iter(x, y, pm.batch_size)
        for x_batch, y_batch in batch_test:
            seq_len = sequence(x_batch)
            feet_dict = self.feed_data(x_batch, y_batch, seq_len, 1.0)
            loss, accuracy = sess.run([self.loss, self.accuracy], feed_dict=feet_dict)
            #y_pred_class, loss, accuracy = sess.run([self.y_pred_cls,self.loss, self.accuracy], feed_dict=feet_dict)
            
            loss_all.append(loss)
            accuracy_all.append(accuracy)

        #return loss, accuracy
        #return y_pred_class,loss, accuracy
        return mean_fun(loss_all), mean_fun(accuracy_all)
    
    
    def getprob(self, sess, x, y):
        label=[]
        pre_label = []
        proba_label = []
        batch_test = batch_iter(x, y, pm.batch_size)
        for x_batch, y_batch in batch_test:
            seq_len = sequence(x_batch)
            feet_dict = self.feed_data(x_batch, y_batch, seq_len, 1.0)
            proba_label_, pre_label_ = sess.run([self.probability, self.predict],feed_dict=feet_dict)
            label.extend(y_batch) 
            proba_label.extend(proba_label_)
            pre_label.extend(pre_label_)

        return label, proba_label, pre_label
    



