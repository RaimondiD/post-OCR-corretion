# post-OCR corretion
A Unimi project that uses NLP tecniques with the aim to correct post OCR errors 

I use the dataset provided for the ALTA 2017 Challenge, it is available on kaggle at the link https://www.kaggle.com/datasets/dmollaaliod/correct-ocr-errors. To run the code create a directory dataset on the main path and put all the files provided for the challenge here.

The main file contains the functions to call to train and execute the model. After seq2seq and BERT are trained, call correct_sentences to test the model on the whole post-OCR problem.
The environment.yml contains all the dependencies used in the project. 
The idea behind the work is in the report.pdf file
