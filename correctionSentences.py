from __future__ import annotations
import nltk
import numpy as np
import modelArgumentManagement
import seq2seqPreprocessing
import itertools
import re
import string
import random
from collections  import defaultdict
from datasets.utils.logging import disable_progress_bar, enable_progress_bar
from BERTEvaluator import BERTEvaluator
from seq2seqEvaluation import avg_levenshtein_ratio

from datasets import Dataset
PROBABILITY_OF_OPERATION = {
    "substitution" : 0.7,
    "deletion" : 0.15,
    "insertion" : 0.15
}
#nltk.download('words') at the first execution download the packa

REGEX_SPECIAL_CHARACTER_MAPPING = [ (r"\?" , "E"), (r"\^" , "S"), (r"\."  , "P"), (r"\\w" , "W"), (r"\\." , "P"), (r"\[" , "Q"), (r"\*", "A"), (r"\(", "T")]
SPECIAL_REGEX_CHARACTER_MAPPING = [ ("E" , r"\?"), ("S" , r"\^"), ("P" , r"\\."), ("W" , r"\\w"), ("Q", r"\["), ("A", r"\*"), ("T", r"\(") ]
EVOLUTIONARY_ARGUMENT_PATH = "evolutionary_argument.json"
    
class Dictionary():
    def __init__(self, train_dataset: list[str]):
        train_words = Dictionary._get_train_word(train_dataset)
        english_words = set(nltk.corpus.words.words())
        #self.set_of_word = get_english_words.union(train_words)
        self.word_datasets = self._get_optimized_dataset(train_words, english_words)
        
    def _get_train_word(train_sentences : list[str]):
        clean_train_sentences = seq2seqPreprocessing.clean_dataset(train_sentences)
        train_words_lists = list(map(lambda sentence : sentence.split(" "), clean_train_sentences))
        flattened_word_list = itertools.chain(*train_words_lists)
        words_without_puntaction = map(remove_punctuaction, flattened_word_list)
        lowercase_words  = [word.lower() for word in words_without_puntaction if len(word)!=0]
        return set(lowercase_words)
    
    def _get_optimized_dataset(self, train_words: set[str], english_words : set[str]) -> Dataset  :
        set_of_words = english_words.union(train_words)
        alphabetical_dict = defaultdict(lambda : [])
        for word in set_of_words:
            alphabetical_dict[word[0]]  = alphabetical_dict[word[0]] + [word]
        for starting_letter in alphabetical_dict:
            alphabetical_dict[starting_letter] = Dictionary._get_dataset_structure(alphabetical_dict[starting_letter])
        return alphabetical_dict
    
    def _get_dataset_structure(list_of_word : list[str]) -> Dataset:
        dataset_formatting = [{"value" : word } for word in list_of_word]
        return Dataset.from_list(dataset_formatting)
        
    def dictionary_matches(self, pattern : str) -> list[str]:
        compiled_pattern = re.compile(r"^" + pattern + r"$") #get the word of the same lenght that match the word
        correct_letters_dataset = self.get_related_datasets(pattern[0])
        if correct_letters_dataset != [[]]:
            disable_progress_bar()
            matches_of_all_datasets = [Dictionary._get_dataset_match(dataset, compiled_pattern) for dataset in correct_letters_dataset]
            enable_progress_bar() #disable and re-enable progress bar
            return set(itertools.chain(*matches_of_all_datasets))
        else:
            return set()
        
    def get_related_datasets(self, first_pattern_letters : str) -> list[Dataset]:  #i'v created a datasets for each starting letters; here i return only the dataset
        if first_pattern_letters == r"\\w":                                                                        #with the first letters that match with the first letter of the pattern
            dictionary_first_letters = [letter for letter in self.word_datasets]
        else:
            dictionary_first_letters = [first_pattern_letters]
        return [self.word_datasets[letter] for letter in dictionary_first_letters]
        
    def _get_dataset_match(targt_dataset : Dataset, pattern : re.compile):
        partial_application_of_filter_function = lambda batch : Dictionary._batch_filter_function(batch, pattern)
        resulting_dataset = targt_dataset.filter(partial_application_of_filter_function, batched=True)
        return [elem['value'] for elem in resulting_dataset.to_list()]
    
    
    def _batch_filter_function(batch : dict[str,list], pattern : re.Pattern):
        is_word_matching = lambda word_entry : re.search(pattern, word_entry)
        return [is_word_matching(word) for word in batch["value"]]


def remove_punctuaction(word : str):
    try:
        if word[-1] in string.punctuation:
            word =  re.sub(r'(.+)[.,!?;]', r'\1', word)
    finally: 
        return word
    
def add_last_punctuation(word : str):
    return word[-1] if re.search(r'[.,!?;]$',word) else ""
    
class WordPattern():
    def __init__(self, pattern : str, score  = 1):
        self.pattern = pattern 
        self.score = score
        
    def get_1_levensthein_distance_words(self, dictionary : Dictionary) -> list[tuple[str,float]]:
        lists_of_patterns = self.get_1_levenshtein_distance_patterns()
        lists_of_lists_of_dictionary_matched_words = list(map(lambda pattern : WordPattern._get_matched_word(pattern,dictionary) , lists_of_patterns))
        dictionary_matched_words =  set(itertools.chain(*lists_of_lists_of_dictionary_matched_words))
        return dictionary_matched_words
    
    def _get_matched_word(pattern : WordPattern, dictionary : Dictionary):
        list_of_matches = dictionary.dictionary_matches(pattern.pattern) 
        return [(match,pattern.score) for match in list_of_matches]      
        
        
    def get_my_matches(self, dictionary: Dictionary):
        return WordPattern._get_matched_word(self, dictionary)

    def get_1_levenshtein_distance_patterns(self) :
        WordPattern.substitute_symbol(self, REGEX_SPECIAL_CHARACTER_MAPPING)
        list_of_match_pattern = []
        list_of_match_pattern += self.generate_substitution_pattern()
        list_of_match_pattern += self.generate_deletion_pattern()
        list_of_match_pattern += self.generate_insertion_pattern()
        word_with_specal_symbol = map(lambda word : WordPattern.substitute_symbol(word, SPECIAL_REGEX_CHARACTER_MAPPING),list_of_match_pattern)
        return set(word_with_specal_symbol)
    
    def generate_substitution_pattern(self):
        return [WordPattern(self.pattern[:i] + r"\w" + self.pattern[i+1:],
                self.score * PROBABILITY_OF_OPERATION["substitution"])
                for i in range(len(self.pattern))]
    
    
    def generate_deletion_pattern(self):
        return [WordPattern(self.pattern[:i] + self.pattern[i+1:],
                              self.score * PROBABILITY_OF_OPERATION["deletion"])
                for i in range(len(self.pattern))]
    
    def generate_insertion_pattern(self):
         return [WordPattern(self.pattern[:i] + r"\w" + self.pattern[i:],
                self.score * PROBABILITY_OF_OPERATION["insertion"])
                for i in range(len(self.pattern))]
    
    
    def substitute_symbol(word, substitutes_list : list[str]):
        for string_symbol, substitute_symbol in substitutes_list:
            word.pattern = re.sub(string_symbol, substitute_symbol, word.pattern)
        return word

class EvolutionaryParameters(modelArgumentManagement.ArgumentFromJson):
    def __init__(self, parameters_file_path = EVOLUTIONARY_ARGUMENT_PATH):
        super().__init__(parameters_file_path)
    
    def get_token_threshold(self):
        return self.parameters_dict['error_treshold']
    
    def get_max_lev_distance(self, word):
        return max(1, min(2, len(word)/2))
    
    def get_number_of_words(self):
        return self.parameters_dict['words_for_generation']
    
    def get_number_of_sentences(self):
        return self.parameters_dict['sentences_for_generation']

class SentenceEvaluator():
    def __init__(self, bert_Evaluator : BERTEvaluator, evolutionary_parameters:EvolutionaryParameters) -> None:
        self.bert_evaluator = bert_Evaluator
        self.evolutionary_parameters = evolutionary_parameters
    
    def get_score(self, text : list[str]):
        return self.bert_evaluator.get_score(text)
    
    def get_max_lev_distance(self, word : str):
        return self.evolutionary_parameters.get_max_lev_distance(word)
    
    def get_number_of_words(self):
        return self.evolutionary_parameters.get_number_of_words()
    
    def get_number_of_sentences(self):
        return self.evolutionary_parameters.get_number_of_sentences()
    
class Sentence():
    def __init__(self, sentence : list[str], sentence_evaluator : SentenceEvaluator) -> None:
        self.text = sentence
        self.sentence_evaluator = sentence_evaluator

    def get_text(self):
        return "".join(self.text)

    def get_score(self):
        return self.sentence_evaluator.get_score(self.text)
    
    def get_error(self) -> list[tuple[str,float]]:
        return self.sentence_evaluator.bert_evaluator.get_wrong_indexes(self.text)
    
    def get_alternative(self, error_tuple : tuple[int,str], alternative_words : list[tuple[int,float]]) -> list[Sentence]: 
        index_error, word_error = error_tuple
        candidate_sentences = self.get_new_sentences(alternative_words, index_error, word_error)
        return self.selected_sentences(candidate_sentences)
        
    
    
    def _get_n_lenveshtein_distance(word_pattern : set[WordPattern]) -> set[WordPattern]:
        word_patterns_lists = [word_pattern.get_1_levenshtein_distance_patterns() for word_pattern in word_pattern]
        return set(itertools.chain(*word_patterns_lists))

    def _get_candidate_words(words_pattern : set[WordPattern], dictionary : Dictionary) -> list[tuple[str,float]]:
        lists_of_lists_of_words =  [pattern.get_my_matches(dictionary) for pattern in words_pattern if pattern.pattern != '']
        return set(itertools.chain(*lists_of_lists_of_words))
    
        
    def get_new_sentences(self, alternative_word : list[tuple[str,float]], index : int, word_error : str) -> list[tuple[Sentence,float]]: 
        new_sentences = [self._generate_sentences(index, candidate) for candidate in alternative_word]
        eventual_puntaction = add_last_punctuation(word_error)
        if(eventual_puntaction):
            alternative_word_with_punctation = [(candidate_word + eventual_puntaction , candidate_word_probability)
                                                 for candidate_word, candidate_word_probability in alternative_word]
            new_sentences = new_sentences + [self._generate_sentences(index, candidate_with_punctation) 
                                 for candidate_with_punctation in alternative_word_with_punctation ]
        return new_sentences
    
    def _generate_sentences(self, index: int, candidate : tuple[str, float]):
        candidate_word, candidate_word_probability = candidate
        return (Sentence(self.text[:index] + [candidate_word] + self.text[index+1:], self.sentence_evaluator), candidate_word_probability)

    def selected_sentences(self, candidate_sentences : list[tuple[Sentence,float]]) -> list[Sentence]:
        sentence_with_score = self.compute_sentence_score(candidate_sentences)
        selection_algorithm = RouletteWheel(self.sentence_evaluator.get_number_of_sentences(), sentence_with_score, lambda tuple : tuple[1])
        return [sentence[0] for sentence, _ in selection_algorithm.extract_individuals()]
    
    def compute_sentence_score(self, generated_sentences : list[tuple[Sentence, float]]):
        sum_of_score = sum([score for _,score in generated_sentences])
        normalized_words_score = [(score/sum_of_score) for _,score in generated_sentences]
        sentences_score = [sentence.get_score() for sentence,_ in generated_sentences]
        sum_of_sentences_score = sum(sentences_score)
        normaized_sentences_score = [score /sum_of_sentences_score for score in sentences_score]
        total_sentence_score = list_sum(normalized_words_score,normaized_sentences_score)
        return [(sentence, score) for sentence, score in zip(generated_sentences, total_sentence_score)]
    
    
    
class BERTSentenceCorrector():
    def __init__(self, dictionary_dataset : list[str], input_dataset : list[str]) -> None:
        self.evolutionary_parameters = EvolutionaryParameters()
        self.dictionary = Dictionary(dictionary_dataset)
        self.sentence_evaluator = SentenceEvaluator(BERTEvaluator(), self.evolutionary_parameters)
        self.input_dataset = input_dataset 
         
    def correct_sentences(self) -> list[str]:
        return [self.sentence_correction(sentence).get_text() for sentence in self.input_dataset]            
        
    def sentence_correction(self, text :str):
        sentence = Sentence(text.split(),self.sentence_evaluator)
        all_sentences = [sentence]
        for error in sentence.get_error():
            selected_sentences = select_best_sentences(all_sentences, self.evolutionary_parameters.get_number_of_sentences())
            alternative_words = self.get_alternative_word(error[1])
            all_sentences = get_candidate_sentences(selected_sentences, error, alternative_words)
        return get_best_sentence(all_sentences)

    def get_alternative_word(self, word : str) -> list[tuple[str,float]]:
            word_without_punctation = remove_punctuaction(word)
            set_of_candidate_pattern = set([WordPattern(word), WordPattern(word_without_punctation)]) #we use set to remove word_without_puncation #
                                                                                                    #if the word and the word without punctatio are equal
            candidate_words = set([(word,1), (word_without_punctation,1)])
            levenshtein_distance = 0
            while(not_enough_generated_word(levenshtein_distance, len(candidate_words), word, self.sentence_evaluator)):
                set_of_candidate_pattern = set_of_candidate_pattern.union(Sentence._get_n_lenveshtein_distance(set_of_candidate_pattern))
                candidate_words = candidate_words.union(Sentence._get_candidate_words(set_of_candidate_pattern, self.dictionary))
                levenshtein_distance +=1
            return candidate_words
    
    def levensthein_distance_improvement(self, ground_truth : list[str]):
         return (avg_levenshtein_ratio(self.input_dataset,ground_truth), avg_levenshtein_ratio(self.correct_sentences(),ground_truth))

def not_enough_generated_word(distance : int, number_of_candidate_words : int, word : str, sentence_evaluator : SentenceEvaluator):
    return (sentence_evaluator.get_max_lev_distance(word) > distance and 
            sentence_evaluator.get_number_of_words() > number_of_candidate_words )

def get_candidate_sentences(list_of_sentences : list[Sentence], error : tuple[str,float], candidate_words : list[tuple[str,float]]) -> list[Sentence]:
    candidate_sentences = [sentence.get_alternative(error, candidate_words) for sentence in list_of_sentences]
    return list(itertools.chain(*candidate_sentences))

def select_best_sentences(all_sentences : list[Sentence], number_of_sentences_to_select : int):
    eval_function = lambda sentence  : sentence.get_score()
    best_sentences = RouletteWheel(number_of_sentences_to_select, all_sentences, eval_function).extract_individuals()
    return best_sentences

def get_best_sentence(selected_sentences : list[Sentence]):
    return max(selected_sentences, key= lambda sentence : sentence.get_score())

class RouletteWheel():
    def __init__(self,number_of_elements : int, sample : list, eval_function : callable) -> None:
        self.number_of_elements = number_of_elements
        self.sample = sorted(sample[:],key = eval_function, reverse= True)
        self.eval_function = eval_function
            
    def extract_individuals(self):
        selected_sample = []
        while (len(selected_sample) < self.number_of_elements and self.sample != []):
            sample_with_probabilistic_metric = self.normalize_sample()
            cumulative_distribution = RouletteWheel._build_cumulative(sample_with_probabilistic_metric)
            extracted_element = RouletteWheel._roulette_spin(cumulative_distribution)
            selected_sample += [extracted_element]
            self.sample.remove(extracted_element)
        return selected_sample
    
    def normalize_sample(self):
        value_of_sample = [self.eval_function(element) for element in self.sample]
        total_value = sum(value_of_sample)
        return [(value/total_value, sample) for value, sample in zip(value_of_sample, self.sample)]
    
    def _build_cumulative(normalizad_sample : list[float,object]):
        cumulative_distribution = []
        sum = 0
        for value,object in normalizad_sample:
            sum += value
            cumulative_distribution.append((sum,object))
        return cumulative_distribution
            
    def _roulette_spin(normalized_sample : list[float,object]):
        random_value = random.random()
        for value, object in normalized_sample:
            if value > random_value:
                return object
            
            
def list_sum(first_list : list[int|float], second_list : list[int|float]) -> list[int|float]:
    assert(len(first_list) == len(second_list))
    sum_of_np_array = np.array(first_list) + np.array(second_list)
    return sum_of_np_array.tolist()