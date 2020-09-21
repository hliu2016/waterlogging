import sys
sys.path.append('E:/Easy_TextCnn_Rnn-master1/TextRnn')
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from Parameters_rnn import parameters as pm
from data_processing_rnn import read_category, get_wordid, get_word2vec, process, batch_iter, sequence
from data_processing_rnn import built_vocab_vector
from Text_Rnn import RnnModel
from sklearn.metrics import roc_auc_score
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.datasets import make_classification
from imblearn.datasets import make_imbalance
#from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.tensorflow import balanced_batch_generator

def getAUC(proba, label):
    global AUC
    pre_prob = []
    for i in range(len(proba)):
        pre_prob.append(proba[i][1])
    AUC = roc_auc_score(label, pre_prob)
    return AUC

def train():

    tensorboard_dir = 'E:/Easy_TextCnn_Rnn-master1/tensorboard/Text_Rnn'
    save_dir = 'E:/Easy_TextCnn_Rnn-master1/checkpoints/Text_Rnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'best_validation')

    tf.summary.scalar('loss', model.loss)
    tf.summary.scalar('accuracy', model.accuracy)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)
    saver = tf.train.Saver()
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    x_train, y_train = process(pm.train_filename, wordid, cat_to_id, max_length=250)  #x_train是标签，y_train是文本数据
    #x_test, y_test = process(pm.test_filename, wordid, cat_to_id, max_length=250)
    val_x, val_y = process(pm.val_filename, wordid, cat_to_id, max_length=250)
    '''
    x_train, y_train = make_classification()
    #通过设置RandomUnderSampler中的replacement=True参数，可以实现自助法（boostrap）抽样
    #通过设置RandomUnderSampler中的ratio参数，可以设置数据采样比例
    rus = RandomUnderSampler(random_state = 0,replacement = True,sampling_strategy = {0:4251,1:4251})#采用随机欠采样（下采样）random_state = 0,sampling_strategy = 0.2
    #pipe = make_pipeline(
            #SMOTE(sampling_strategy = {0:4250}),
            #NearMiss(sampling_strategy = {1:4250})
                         #)
    #x_resample,y_resample=pipe.fit_resample(x_train,y_train)
    x_resample,y_resample=rus.fit_sample(x_train,y_train)
    '''
    
    
    '''
    data1=x_train[x_train['label']=='负']
    data0=x_train[x_train['label']=='正']
    index = np.random.randint(
            len(data1),size=1*(len(x_train)-len(data1)))
    lower_data1 = data1.iloc[list(index)]#下采样
    '''
    '''
    ratio = {0:4251,1:4251}
    x_imb,y_imb = make_imbalance(x_train, y_train,ratio=ratio)
    #x_imb = np.array(x_imb)
    #y_imb = np.array(y_imb)
    '''
    '''
    model_RandomUnderSampler = RandomUnderSampler()
    x_RandomUnderSample_resampled,y_RandomUnderSample_resampled = model_RandomUnderSampler.fit_sample(x_train, y_train)
    #RandomUnderSampler_resampled=pd.concat([x_RandomUnderSample_resampled,y_RandomUnderSample_resampled],axis=1)
    '''    
    '''
    x_train, y_train = make_classification()
    ee=EasyEnsembleClassifier(random_state=0,sampling_strategy='majority')#sampling_strategy=0.2
    #x_resampled,y_resampled == ee.fit_sample(x_train, y_train)
    #ee.fit(x_train, y_train)
    x_train,y_train == ee.fit(x_train, y_train)
    '''
    
    
    #class_dict = dict()
    #class_dict[0] = 4251;class_dict[1] = 4251
    #x_train,y_train = make_imbalance(x_train,y_train,class_dict)
    #0表示正样本，1表示负样本
    training_generator,steps_per_epoch = balanced_batch_generator(
            x_train,y_train,sampler=RandomUnderSampler(sampling_strategy={0:4251,1:4251}),batch_size=pm.batch_size,random_state=42
            )#sample_weight = None,
    for epoch in range(pm.num_epochs):
        print('Epoch:', epoch+1)
        #num_batchs = int((len(x_train) - 1) / pm.batch_size) + 1
        for i in range(steps_per_epoch):
            x_batch,y_batch=next(training_generator)
            #feed_dict = dict(y_batch,x_batch)
            seq_len = sequence(x_batch)
            feed_dict = model.feed_data(x_batch, y_batch, seq_len, pm.keep_prob)
            #feed_dict[pm.input_y]=y_batch;feed_dict[targets] = x_batch
            _, global_step, _summary, train_loss, train_accuracy = session.run([model.optimizer, model.global_step, merged_summary,
                                                                    model.loss, model.accuracy],feed_dict=feed_dict)
            if global_step % 100 == 0:
                #test_loss, test_accuracy = model.evaluate(session, x_test, y_test)
                val_loss, val_accuracy = model.evaluate(session, val_x, val_y)
                print('global_step:', global_step, 'train_loss:', train_loss, 'train_accuracy:', train_accuracy,
                      #'test_loss:', test_loss, 'test_accuracy:', test_accuracy)
                      'val_loss:', val_loss, 'val_accuracy:', val_accuracy)
                #label, proba_label, pre_label = model.getprob(session, x_test, y_test)
                label, proba_label, pre_label = model.getprob(session, val_x, val_y)
                label = np.argmax(label, 1).tolist()
                AUC = getAUC(proba_label, label)
#                print(np.argmax(label, 1).tolist()[:10])
#                print(proba_label[:10])
#                print(pre_label[:10])
                               
                ACC, SN, SP, Precision, F1, MCC,TP,FN,FP,TN = performance(label,pre_label)##
                #print('ACC:%.3f SN:%.3f SP:%.3f Precision:%.3f F1:%.3f MCC:%.3f AUC:%.3f' %(ACC, SN, SP, Precision, F1, MCC, AUC))##
                print('ACC:%.3f SN:%.3f SP:%.3f Precision:%.3f F1:%.3f MCC:%.3f AUC:%.3f TP:%d,FN:%d,FP:%d,TN:%d' %(ACC, SN, SP, Precision, F1, MCC, AUC,TP,FN,FP,TN))
                #print('test_AUC:', AUC)
                print('val_AUC:', AUC)
                #print('Saving Model...')
                #saver.save(session, save_path, global_step=global_step)
            if global_step % steps_per_epoch == 0:
            #if global_step % num_batchs == 0:
                pre_=[]
                for i in range(0,400):
                    pre_.append(proba_label[i][1])                    
                #np.savetxt(r'E:\Easy_TextCnn_Rnn-master1\TextRnn'+"/scores-val-1.data",pre_,fmt="%f",delimiter="\t")
                print('Saving Model...')
                saver.save(session, save_path, global_step=global_step)

        pm.learning_rate *= pm.lr_decay
    
    '''
    for epoch in range(pm.num_epochs):
        print('Epoch:', epoch+1)
        num_batchs = int((len(x_train) - 1) / pm.batch_size) + 1
        batch_train = batch_iter(x_train, y_train, batch_size=pm.batch_size)
        #batch_train = batch_iter(x_resample, y_resample, batch_size=pm.batch_size)
        #batch_train = batch_iter(lower_data1, y_train, batch_size=pm.batch_size)
        #batch_train = batch_iter(x_imb,y_imb, batch_size=pm.batch_size)
        #batch_train = batch_iter(X_resampled,y_resampled, batch_size=pm.batch_size)
        #batch_train = batch_iter(x_RandomUnderSample_resampled,y_RandomUnderSample_resampled, batch_size=pm.batch_size)
        for x_batch, y_batch in batch_train:
            seq_len = sequence(x_batch)
            feed_dict = model.feed_data(x_batch, y_batch, seq_len, pm.keep_prob)
            _, global_step, _summary, train_loss, train_accuracy = session.run([model.optimizer, model.global_step, merged_summary,
                                                                                model.loss, model.accuracy],feed_dict=feed_dict)
            if global_step % 100 == 0:
                test_loss, test_accuracy = model.evaluate(session, x_test, y_test)
                print('global_step:', global_step, 'train_loss:', train_loss, 'train_accuracy:', train_accuracy,
                      'test_loss:', test_loss, 'test_accuracy:', test_accuracy)
                
                
                
                label, proba_label, pre_label = model.getprob(session, x_test, y_test)
                label = np.argmax(label, 1).tolist()
                AUC = getAUC(proba_label, label)
#                print(np.argmax(label, 1).tolist()[:10])
#                print(proba_label[:10])
#                print(pre_label[:10])
                
                #pre_=[]
                #for i in range(0,400):
                    #pre_.append(proba_label[i][1])
                #np.savetxt(r'E:\Easy_TextCnn_Rnn-master1\TextRnn'+"/scores_output.data",pre_,fmt="%f",delimiter="\t")
                
                ACC, SN, SP, Precision, F1, MCC = performance(label,pre_label)##
                print('ACC:%.3f SN:%.3f SP:%.3f Precision:%.3f F1:%.3f MCC:%.3f AUC:%.3f' %(ACC, SN, SP, Precision, F1, MCC, AUC))##
                
                print('test_AUC:', AUC)
                
                

            if global_step % num_batchs == 0:
                print('Saving Model...')
                saver.save(session, save_path, global_step=global_step)

        pm.learning_rate *= pm.lr_decay
    '''    




#def performance(labelArr, predictArr):
def performance(label, pre_label):
    TP = 0.; TN = 0.; FP = 0.; FN = 0.   
    for i in range(len(label)):
        if label[i] == 1 and pre_label[i] == 1:
            TP += 1.
        if label[i] == 1 and pre_label[i] == 0:
            FN += 1.
        if label[i] == 0 and pre_label[i] == 1:
            FP += 1.
        if label[i] == 0 and pre_label[i] == 0:
            TN += 1.
    ACC = (TP + TN) / (TP + FN + FP + TN)
    SN = TP/(TP + FN) 
    SP = TN/(FP + TN) 
    if TP + FP != 0:       
        Precision = TP/(TP + FP)
    else:
        Precision = 0
    Recall = SN
    if Precision + Recall != 0:        
        F1 = 2 * (Precision * Recall)/(Precision + Recall)
    else:
        F1 = 0
    fz = float(TP*TN - FP*FN)
    fm = float(np.math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
    if fm !=0:        
        MCC = fz/fm
    else:
        MCC = 0    
    return ACC, SN, SP, Precision, F1, MCC,TP,FN,FP,TN
'''
def mean_fun(onelist):
    count = 0
    for i in onelist:
        count += i
    return float(count/len(onelist))
'''







if __name__ == '__main__':

    pm = pm
    filenames = [pm.train_filename, pm.test_filename, pm.val_filename]
    categories, cat_to_id = read_category()
    wordid = get_wordid(pm.vocab_filename)
    #pm.vocab_size = len(wordid)
    pm.pre_trianing = get_word2vec(pm.vector_word_npz)

    model = RnnModel()

    train()
    
    #val()#
    
   
    
    '''
    sentences = []
    label2 = []
    #categories, cat_to_id = read_category()
    #wordid = get_wordid(pm.vocab_filename)
    #pm.vocab_size = len(wordid)
    #pm.pre_trianing = get_word2vec(pm.vector_word_npz)
    
    #model = RnnModel()

    pre_label, label = val()

    correct = np.equal(pre_label, np.argmax(label, 1))
    accuracy = np.mean(np.cast['float32'](correct))
    print('accuracy:', accuracy)
    print('ACC:%.3f SN:%.3f SP:%.3f Precision:%.3f F1:%.3f MCC:%.3f AUC:%.3f' %(ACC, SN, SP, Precision, F1, MCC, AUC))
    '''
    
    
    

  