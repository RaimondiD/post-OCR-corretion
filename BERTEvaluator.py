from transformers import pipeline , AutoModelForTokenClassification, AutoTokenizer
import torch
from functools import reduce
from itertools import groupby

MODEL_PATH = "models/finetuned_bert/checkpoint-1315"
class BERTEvaluator():
    def __init__(self):
        self.model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    def evalate_phrase(self, text : list[str])-> list[str,float]:
        inputs = self.tokenizer(text, return_tensors="pt", is_split_into_words= True)
        token_word_mapping = inputs.word_ids()
        with torch.no_grad():
            logits = self.model(**inputs).logits 
        list_of_classes_probability = logits.softmax(dim=-1).tolist()[0]  #the [0] is used to remove the most esternal class that 
        correct_probability = [prob_correct for _, prob_correct in list_of_classes_probability]
        probability_word_mapping = [(word_idx,probability) for word_idx, probability in zip(token_word_mapping, correct_probability)if word_idx != None]
        probability_grouped_for_word =  groupby(probability_word_mapping, key= lambda elem : elem[0])
        words_probability = [(text[idx], min(probabilities)[1]) for idx, probabilities in probability_grouped_for_word]
        return words_probability

    def get_wrong_indexes(self, text : list[str]):
        return [(index, word) for index, (word, probability) in enumerate(self.evalate_phrase(text)) if probability < 0.5]
    
    def get_score(self, text : list[str]):
        probability_of_word = self.evalate_phrase(text)
        sum_of_probability = sum([word_and_probability[1] for word_and_probability in probability_of_word])
        return sum_of_probability/len(probability_of_word)