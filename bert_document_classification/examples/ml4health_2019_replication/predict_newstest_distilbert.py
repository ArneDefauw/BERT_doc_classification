import sys, os, logging, torch

#appends current directory to sys path allowing data imports.
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

sys.path.append(  "/notebook/nas-trainings/arne/OCCAM/text_classification_BERT/code_BERT/bert_document_classification"  )

from data import load_newstest, generator_newstest
from sklearn.metrics import f1_score, precision_score, recall_score
from bert_document_classification.models import NewstestDistilBert

log = logging.getLogger()

if __name__ == "__main__":

    newstest_bert = NewstestDistilBert(model_name="/notebook/nas-trainings/arne/OCCAM/text_classification_BERT/code_BERT/bert_document_classification/examples/ml4health_2019_replication/results_newstest_single_gpu_lstm_batch_size10_unfreeze_last_layers_distillbert/run_2020_03_16_13_28_02_e36ea6cdbe42/checkpoint_500", device='cuda:1') #CPU prediction (change to 'cuda' if possible)
    
    labels = newstest_bert.labels
    
    _ , data_test, target_names = load_newstest( labels )
    
    test=list(generator_newstest( data_test, target_names ))
    
    #test=test[:120]
    
    test_documents, test_labels = [],[]
    for _, text, status in test:
        test_documents.append(text)
        label = [0]*len(labels)
        for idx, name in enumerate(labels):
            if name == status:
                label[idx] = 1
        test_labels.append(label)
        
       
    #statistics test corpus
    #import numpy as np
    #length=[]
    #for idx, doc in enumerate( test_documents ):
    #    length.append( len(doc.split()) )
    #    print( idx, len(doc.split()) )
    #print( "mean",  np.mean(length) )
    #print( "std",  np.std(length) )
    #print( "max",  np.max(length) )
    #print( "min",  np.min(length) )


    correct_labels = torch.FloatTensor(test_labels).transpose(0,1)

    predictions = newstest_bert.predict(test_documents)
    assert correct_labels.shape == predictions.shape

    precisions = []
    recalls = []
    fmeasures = []
    for label_idx in range(predictions.shape[0]):
        correct = correct_labels[label_idx].view(-1).numpy()
        predicted = predictions[label_idx].view(-1).numpy()
        present_f1_score = f1_score(correct, predicted, average='binary', pos_label=1)
        present_precision_score = precision_score(correct, predicted, average='binary', pos_label=1)
        present_recall_score = recall_score(correct, predicted, average='binary', pos_label=1)

        precisions.append(present_precision_score)
        recalls.append(present_recall_score)
        fmeasures.append(present_f1_score)


    print('Metric\t' + '\t'.join([labels[label_idx] for label_idx in range(predictions.shape[0])]))
    print('Precision\t' + '\t'.join([str(precisions[label_idx]) for label_idx in range(predictions.shape[0])]))
    print('Recall\t' + '\t'.join([str(recalls[label_idx]) for label_idx in range(predictions.shape[0])]))
    print('F1\t' + '\t'.join([str(fmeasures[label_idx]) for label_idx in range(predictions.shape[0])]))
    print('Micro F1\t' + str(f1_score(correct_labels.reshape(-1).numpy(), predictions.reshape(-1).numpy(), average='micro')) )
    print('Macro F1\t' + str(f1_score(correct_labels.reshape(-1).numpy(), predictions.reshape(-1).numpy(), average='macro')) )