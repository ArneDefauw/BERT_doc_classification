# BERT_doc_classification
Document classification with BERT

Code based on https://github.com/AndriyMulyar/bert_document_classification.

With some modifications:

-switch from the pytorch-transformers to the transformers ( https://github.com/huggingface/transformers ) library.
-unfreezing of the last layers of BERT, instead of freezing complete BERT (the latter results in subpar peformance).
-support for document classification using DistilBert ( BERT with 40% less parameters )

Information on how to get data to try replicate results reported in (https://arxiv.org/abs/1910.13664 ), see. 
https://github.com/ArneDefauw/bert_document_classification/blob/master/examples/ml4health_2019_replication/data/README.md

In short, you must sign the appropriate agreements:
https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/

Afterwards, you can download the data. Folder structure should look like this.



```bash
examples/ml4health_2019_replication/data/
├── obesity_patient_records_test.xml
├── obesity_patient_records_training2.xml
├── obesity_patient_records_training.xml
├── obesity_standoff_annotations_test.xml
├── obesity_standoff_annotations_training_addendum2.xml
├── obesity_standoff_annotations_training_addendum3.xml
├── obesity_standoff_annotations_training_addendum.xml
├── obesity_standoff_annotations_training.xml
├── README.md
├── smokers_surrogate_test_all_version2.xml
└── smokers_surrogate_train_all_version2.xml

```


#### Make sure you have downloaded ClinicalBERT and referenced it's path in each respective config file!
Clinical Bert: https://github.com/EmilyAlsentzer/clinicalBERT


#### For training of Bert for Document classification: 

Use the Config files in:
https://github.com/ArneDefauw/BERT_doc_classification/blob/master/bert_document_classification/examples/ml4health_2019_replication

And the python scripts ( 	train_n2c2_2006.py, train_n2c2_2008.py, train_newstest.py ). The latter script will train a system for document classification on the Newsgroup dataset (https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html). The Newsgroup dataset will be automatically downloaded when running the script. 


#### For evaluation:

Use the predict_n2c2_2006.py, predict_n2c2_2008.py, predict_newstest_bert.py, predict_newstest_distilbert.py,


#### Architectures:

This repository supports 5 architectures: DocumentBertLSTM, DocumentDistilBertLSTM, DocumentBertTransformer, DocumentBertLinear, DocumentBertMaxPool. 


## Note: none of these architectures could replicate results reported in https://arxiv.org/abs/1910.13664.

### Results on n2c2_2006 task:

Freezing Clinical BERT:

https://github.com/ArneDefauw/BERT_doc_classification/tree/master/bert_document_classification/examples/ml4health_2019_replication/results_single_gpu_single_lstm_batch_size10/run_2020_03_04_10_37_12_58d26bafbcb2

Unfreezing last encoding layer of Clinical BERT:

https://github.com/ArneDefauw/BERT_doc_classification/tree/master/bert_document_classification/examples/ml4health_2019_replication/results_single_gpu_single_lstm_batch_size10_unfreeze_last_layers/run_2020_03_11_12_51_11_f3f33a8bb1ff


### Results on n2c2_2008 task:

Unfreezing last encoding layer of Clinical BERT:

https://github.com/ArneDefauw/BERT_doc_classification/tree/master/bert_document_classification/examples/ml4health_2019_replication/results_2008_final_unfreeze_bert_last_layer/run_2020_03_16_15_27_57_e2b11bf5bd3b


### Results on newsgroup task:

Freezing bert-base-uncased:

https://github.com/ArneDefauw/BERT_doc_classification/tree/master/bert_document_classification/examples/ml4health_2019_replication/results_newstest_single_gpu_lstm_batch_size10/run_2020_03_04_12_52_51_58d26bafbcb2

Unfreezing last encoding layer of bert-base-uncased:

https://github.com/ArneDefauw/BERT_doc_classification/tree/master/bert_document_classification/examples/ml4health_2019_replication/results_newstest_single_gpu_lstm_batch_size10_unfreeze_last_layers/run_2020_03_11_13_02_42_f3f33a8bb1ff


Unfreezing last encoding layer of distilbert-base-uncased:

https://github.com/ArneDefauw/BERT_doc_classification/tree/master/bert_document_classification/examples/ml4health_2019_replication/results_newstest_single_gpu_lstm_batch_size10_unfreeze_last_layers_distillbert/run_2020_03_16_13_28_02_e36ea6cdbe42
