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

from datasets import Dataset
PROBABILITY_OF_OPERATION = {
    "substitution" : 0.7,
    "deletion" : 0.15,
    "insertion" : 0.15
}
#nltk.download('words') at the first execution download the packa

REGEX_SPECIAL_CHARACTER_MAPPING = [ (r"\?" , "E"), (r"\^" , "S"), (r"\."  , "P"), (r"\\w" , "W"),(r"\\." , "P")]
SPECIAL_REGEX_CHARACTER_MAPPING = [("E" , r"\?"), ("S" , r"\^"), ("P" , r"\\."), ("W" , r"\\w")]
EVOLUTIONARY_ARGUMENT_PATH = "evolutionary_argument.json"
    
class Dictionary():
    def __init__(self, train_dataset: list[str]):
        train_words = Dictionary._get_train_word(train_dataset)
        english_words = set(nltk.corpus.words.words())
        #self.set_of_word = get_english_words.union(train_words)
        self.word_datasets = self._get_optimized_dataset(train_words, english_words)
        
    def _get_train_word(train_phrases : list[str]):
        clean_train_phrases = seq2seqPreprocessing.clean_dataset(train_phrases)
        train_words_lists = list(map(lambda prhase : prhase.split(" "), clean_train_phrases))
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
        disable_progress_bar()
        matches_of_all_datasets = [Dictionary._get_dataset_match(dataset, compiled_pattern) for dataset in correct_letters_dataset]
        enable_progress_bar() #disable and re-enable progress bar
        return set(itertools.chain(*matches_of_all_datasets))

    def get_related_datasets(self, first_pattern_letters : str) -> list[Dataset]:  #i'v created a datasets for each starting letters; here i return only the dataset
                                                                                   #with the first letters that match with the first letter of the pattern
        pattern_of_first_letters = f"^{first_pattern_letters}$"
        dictionary_first_letters = [letter for letter in self.word_datasets if re.match(pattern_of_first_letters, letter) ]
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
    
    def get_number_of_phrases(self):
        return self.parameters_dict['phrases_for_generation']

class PhraseEvaluator():
    def __init__(self, bert_Evaluator : BERTEvaluator, evolutionary_parameters:EvolutionaryParameters) -> None:
        self.bert_evaluator = bert_Evaluator
        self.evolutionary_parameters = evolutionary_parameters
    
    def get_score(self, text : list[str]):
        return self.bert_evaluator.get_score(text)
    
    def get_max_lev_distance(self, word : str):
        return self.evolutionary_parameters.get_max_lev_distance(word)
    
    def get_number_of_words(self):
        return self.evolutionary_parameters.get_number_of_words()
    
    def get_number_of_phrases(self):
        return self.evolutionary_parameters.get_number_of_phrases()
    
class Phrase():
    def __init__(self, prhase : list[str], phrase_evaluator : PhraseEvaluator) -> None:
        self.text = prhase
        self.phrase_evaluator = phrase_evaluator

    def get_text(self):
        return self.text

    def get_score(self):
        return self.phrase_evaluator.get_score(self.text)
    
    def get_error(self) -> list[tuple[str,float]]:
        return self.phrase_evaluator.bert_evaluator.get_wrong_indexes(self.get_text())
    
    def get_alternative(self, error_tuple : tuple[int,str], dictionary : Dictionary) -> list[Phrase]: 
        index_error, word_error = error_tuple
        alternative_words = self.get_alternative_word(word_error, dictionary)
        candidate_phrases = self.get_new_phrases(alternative_words, index_error, word_error)
        return self.selected_phrases(candidate_phrases)
        
    def get_alternative_word(self, word:str, dictionary : Dictionary) -> list[tuple[str,float]]:
        word_without_punctation = remove_punctuaction(word)
        set_of_candidate_pattern = set([WordPattern(word), WordPattern(word_without_punctation)]) #we use set to remove word_without_puncation #
                                                                                                  #if the word and the word without punctatio are equal
        candidate_words = set([(word,1), (word_without_punctation,1)])
        levenshtein_distance = 0
        while(self._not_enough_generated_word(levenshtein_distance, len(candidate_words), word)):
            set_of_candidate_pattern = set_of_candidate_pattern.union(Phrase._get_n_lenveshtein_distance(set_of_candidate_pattern))
            candidate_words = candidate_words.union(Phrase._get_candidate_words(set_of_candidate_pattern, dictionary))
            levenshtein_distance +=1
        return candidate_words
    
    def _get_n_lenveshtein_distance(word_pattern : set[WordPattern]) -> set[WordPattern]:
        word_patterns_lists = [word_pattern.get_1_levenshtein_distance_patterns() for word_pattern in word_pattern]
        return set(itertools.chain(*word_patterns_lists))

    def _get_candidate_words(words_pattern : set[WordPattern], dictionary : Dictionary) -> list[tuple[str,float]]:
        lists_of_lists_of_words =  [pattern.get_my_matches(dictionary) for pattern in words_pattern]
        return set(itertools.chain(*lists_of_lists_of_words))
    
    def _not_enough_generated_word(self, distance : int, number_of_candidate_words, word:str):
        return (self.phrase_evaluator.get_max_lev_distance(word) > distance and 
                self.phrase_evaluator.get_number_of_words() > number_of_candidate_words )
        
    def get_new_phrases(self, alternative_word : list[tuple[str,float]], index : int, word_error : str) -> list[tuple[Phrase,float]]: 
        new_phrases = [self._generate_phrases(index, candidate) for candidate in alternative_word]
        eventual_puntaction = add_last_punctuation(word_error)
        if(eventual_puntaction):
            alternative_word_with_punctation = [(candidate_word + eventual_puntaction , candidate_word_probability)
                                                 for candidate_word, candidate_word_probability in alternative_word]
            new_phrases = new_phrases + [self._generate_phrases(index, candidate_with_punctation) 
                                 for candidate_with_punctation in alternative_word_with_punctation ]
        return new_phrases
    
    def _generate_phrases(self, index: int, candidate : tuple[str, float]):
        candidate_word, candidate_word_probability = candidate
        return (Phrase(self.text[:index] + [candidate_word] + self.text[index+1:], self.phrase_evaluator), candidate_word_probability)

    def selected_phrases(self, candidate_phrases : list[tuple[Phrase,float]]) -> list[Phrase]:
        prhase_with_score = self.compute_phrase_score(candidate_phrases)
        selection_algorithm = RouletteWheel(self.phrase_evaluator.get_number_of_phrases(), prhase_with_score, lambda tuple : tuple[1])
        return [phrase[0] for phrase, _ in selection_algorithm.extract_individuals()]
    
    def compute_phrase_score(self, generated_phrases : list[tuple[Phrase, float]]):
        sum_of_score = sum([score for _,score in generated_phrases])
        normalized_words_score = [(score/sum_of_score) for _,score in generated_phrases]
        prhases_score = [phrase.get_score() for phrase,_ in generated_phrases]
        sum_of_phrases_score = sum(prhases_score)
        normaized_phrases_score = [score /sum_of_phrases_score for score in prhases_score]
        total_phrase_score = list_sum(normalized_words_score,normaized_phrases_score)
        return [(phrase, score) for phrase, score in zip(generated_phrases, total_phrase_score)]
        
def phrase_correction(text :str, dictionary):
    evolutionary_parameters = EvolutionaryParameters()
    phrase_evaluator = PhraseEvaluator(BERTEvaluator(), evolutionary_parameters)
    phrase = Phrase(text.split(),phrase_evaluator)
    all_phrases = [phrase]
    for error in phrase.get_error():
        selected_phrases = select_best_phrases(all_phrases, evolutionary_parameters.get_number_of_phrases())
        all_phrases = get_candidate_phrases(selected_phrases, error, dictionary)
    return get_best_phrase(all_phrases)

def get_candidate_phrases(list_of_phrases : list[Phrase], error : tuple[str,float], dictionary : Dictionary) -> list[Phrase]:
    candidate_phrases = [phrase.get_alternative(error, dictionary) for phrase in list_of_phrases]
    return list(itertools.chain(*candidate_phrases))

def select_best_phrases(all_phrases : list[Phrase], number_of_phrases_to_select : int):
    eval_function = lambda phrase  : phrase.get_score()
    best_phrases = RouletteWheel(number_of_phrases_to_select, all_phrases, eval_function).extract_individuals()
    return best_phrases

def get_best_phrase(selected_phrases : list[Phrase]):
    return max(selected_phrases, key= lambda prhase : prhase.get_score())

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