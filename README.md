# BERT_doc_classification
Document classification with BERT

Code based on https://github.com/AndriyMulyar/bert_document_classification.

Information on how to get data, see. 
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



