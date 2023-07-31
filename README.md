# post-OCR corretion
 a unimi project that adress the task of experimentig information retrivial tecnique with the aim to correct OCR error

I use the dataset provided for the ALTA 2017 Challenge, it is available on kaggle at the link https://www.kaggle.com/datasets/dmollaaliod/correct-ocr-errors. To run the code create a directory dataset on the main path and put here all the file provided for the challenge.

In the seq2seq files there are the code to build, train and test a sequence2sequence model that given a list of char without space, try to divide it in the correct way, the
idea is to use this approach to resolve the segmentation problem of the OCR system. 

Then i try to use a fine-tuned BERT model to find the wrong words and use the model, a dictionary and the levneshtein distance of the similar word to try to correct the words.
