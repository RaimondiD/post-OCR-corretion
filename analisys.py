import BERTEvaluator
import BERTpreprocessing
import trainTestSplitter
import textSegmentator
import numpy as np
from functools import reduce
import nltk
import itertools
import metrics


class GlobaEvaluator():
    def __init__(self):
        self.dataset_object = trainTestSplitter.TrainTestSplitter()

    def evaluate_Bert_model(self):
        bert_test_dataset = BERTpreprocessing.DatasetPreprocessor(self.dataset_object).get_test_dataset()
        bert_evaluator = BERTEvaluator.BERTEvaluator()
        bert_metrics = bert_evaluator.get_model_metrics(bert_test_dataset)
        GlobaEvaluator._print_avg_and_std_deviation("precision", bert_metrics["precision"])
        GlobaEvaluator._print_avg_and_std_deviation("recall", bert_metrics["recall"])
        GlobaEvaluator._print_avg_and_std_deviation("f1-score", bert_metrics["f1-score"])
        GlobaEvaluator._print_avg_and_std_deviation("accuracy", bert_metrics["accuracy"])
        global_confusion_matrix = reduce(
            lambda matrix,another_matrix : matrix + another_matrix, 
            bert_metrics["confusion_matrix"]
            )
        metrics.plot_confusion_matrix(global_confusion_matrix) 

    def _print_avg_and_std_deviation(name_of_metric : str, data : list):
        print(f"average {name_of_metric} : {np.average(np.array(data))} \
              std deviation of {name_of_metric}: {np.std(np.array(data))}")
    
    def dataset_analysis(self):
        output_data = self.dataset_object.get_output_dataset()
        input_data = self.dataset_object.get_input_dataset()
        input_not_in_dict_words, input_words = GlobaEvaluator._dataset_words_info(input_data)
        output_not_in_dict_words, output_words = GlobaEvaluator._dataset_words_info(output_data)
        uncommon_input_words_ratio = input_not_in_dict_words / input_words
        uncommon_output_words_ratio = output_not_in_dict_words / output_words
        approximate_segmentation_ratio = (output_words - input_words)/output_words #assuming that output dosen't have segmenation errors
        approximate_ratio_of_total_errors = GlobaEvaluator._get_total_words_differences(input_data, output_data) /output_words
        print(f"words not in dictionary ratio : \n input : {uncommon_input_words_ratio} \n output : {uncommon_output_words_ratio}")
        print("--------------------------------")
        print(f"approximate ratio of errors : {approximate_ratio_of_total_errors}\n segmentation :{approximate_segmentation_ratio} \n other : {approximate_ratio_of_total_errors - approximate_segmentation_ratio}")

    def _dataset_words_info(data :list[str])->tuple[int,int]:
        data_words = GlobaEvaluator._list_of_words_from_dataset(data)
        english_words = set(nltk.corpus.words.words())
        not_in_dictionary_words = [not_standard_word for not_standard_word in data_words if not_standard_word not in english_words]
        return len(not_in_dictionary_words), len(data_words)
    
    def _get_total_words_differences(input_dataset : list[str], output_dataset : list[str]):
        input_list_of_list_of_words = [phrases.split(" ") for phrases in input_dataset]
        output_list_of_list_of_words = [phrases.split(" ") for phrases in output_dataset]
        total_words_differences = 0
        for phrase_input, phrase_output in zip(input_list_of_list_of_words,output_list_of_list_of_words):
            total_words_differences += len([different_word for different_word in phrase_input if different_word not in phrase_output])
        return total_words_differences

    def _list_of_words_from_dataset(output_datset : list[str]):
        list_of_list_of_words = [phrases.split(" ") for phrases in output_datset]
        list_of_words = list(itertools.chain(*list_of_list_of_words))
        return list_of_words

    def evaluate_seq2seq(self):
        segmentator = textSegmentator.Seq2SeqTextSplitter(self.dataset_object)
        segmentator.evalute_seq2seq()

    
    def global_analysis(self):
        print("dataset analysis: ")
        self.dataset_analysis()
        print("segmenator analysis: ")
        self.evaluate_seq2seq()
        print("Bert metrics :")
        self.evaluate_Bert_model()


evaluator = GlobaEvaluator()
evaluator.evaluate_Bert_model()