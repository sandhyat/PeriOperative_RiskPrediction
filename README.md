This repository contains the source code for various approaches that can used for postoperative complication prediction.

### Overview

In brief, there are two approaches: 
1) prediction classifiers are learnt end to end in a supervised manner using different combination of data modalities, 
2) first a self supervised representation of a data modality such as time series is learnt followed by a supervised classifier on the learnt representation.

In clinical prediction setup, access to the data modalities can vary and need not be available for all the patients. 
Some of the common data modalities observed in Electronic Health Records are as follows: preoperative assessment, free text about surgical procedure, past medical history, home medications and problem list, intraoperative flowsheets, intraoperative medications, ICD codes (postoperatively).
Our task of interest (outcome) is predicting post-operative complications such as 30-day inhospital mortality, unplanned ICU admission, Acute Kidney Injury. These labels are retrospectively created by experts or extracted from the hospital billing data.

Based on varying algorithm complexity the approaches mentioned can be explained below:

1) A prediction classifier that uses the data that is available prior to the start of surgery. As this data exists in tabular form or can be converted in tabular (text embeddings), training a shallow classifier using XGBoost or Logistic Regression is the first and most straightforward approach. Recently, more complex and high capacity learners such as transformers are also available for tabular data. 

2) A prediction classifier that uses all the data that is available until the end of surgery. This method needs special techniques to learn from the intraoperative time series data. One can use LSTMs, attention mechanism or Transformer architecture along with feed forward networks for the tabular data and train them jointly.
![End to End Architecture](/Images/End-toEnd_Supervised.png)

3) A prediction classifier that uses all the data that is available retrospectively (including the outcomes) for self supervised learning (commonly known as pretraining) and using everything before the end of surgery for training the classifiers. One can use many of the techniques currently available for pretraining including contrastive learning for the first stage of representation learning. For the second stage of classifier learning, even training a shallow classifier suffices sometimes.
![MVCL Architecture](/Images/MVCL_SelfSupervised.png)

### Setups

##### 1) Requirements

##### 2) Data format  