import csv
from pathlib import Path
import regex as re
from typing import Callable
import more_itertools
import itertools
from collections import defaultdict

DATASET_DIRCECTORY = Path("dataset")
TEST_INPUT_FILE = Path(DATASET_DIRCECTORY,"train_output.csv")  #I use only the output file because i want to fine-tune a model that learn how to put spaces in text,
#so the input must be a a text without spaces and the outut the text with the spaces in the correct
#position. To build this type of dataset i need a dataset with the spaces in the correct position, so i use the output file."""
ENCODING_DEFAULT_SEPARATOR_SYMBOL = "#"
DATASET_WORD_SEPARATOR_SYMBOL = ' '
MAX_SEQUENCE_LEN = 256 
DEFAULT_UNKNOW_SYMBOL = "<unk>"
END_OF_SENTENCE_SYMBOL = '<eos>'  
START_OF_SENTENCE_SYMBOL = '<bos>'
DEFAULT_PADDING_SYMBOL = '<pad>'


class IndexTranslator():
    
    def __init__(self, dataset : list [list[str]]) -> None:
        self.special_symbol = [DEFAULT_UNKNOW_SYMBOL,DEFAULT_PADDING_SYMBOL,START_OF_SENTENCE_SYMBOL,END_OF_SENTENCE_SYMBOL]
        self.index_char_dictionary = self.__create_char_dictionary(dataset)
        char_index_strict_dictionary = {char:index for index,char in self.index_char_dictionary.items()}
        self.char_index_dictionary = defaultdict(lambda : self.special_symbol.index(DEFAULT_UNKNOW_SYMBOL), char_index_strict_dictionary)
        
    def encode_sequence(self, sentence_as_a_sequence :list[str]) -> list[str]:
        return [self.char_index_dictionary[char] for char in sentence_as_a_sequence]
    
    def sequence_from_encode(self, encoded_sequence :list[int]) -> list[str]:
        sequence_in_natural_language = [self.index_char_dictionary[index] for index in encoded_sequence]
        trunked_sequence_in_natural_language = self._remove_character_after_eos(sequence_in_natural_language)
        sequence_without_special_symbol = [character for character in trunked_sequence_in_natural_language 
                                           if character not in self.special_symbol]
        return sequence_without_special_symbol
    
    def _remove_character_after_eos(self, natural_language_sequence : list[str]):
        index_of_EOS = "".join(natural_language_sequence).find(END_OF_SENTENCE_SYMBOL)
        return natural_language_sequence[:index_of_EOS]
        
    
    def __create_char_dictionary(self,dataset : list [list[str]]):
        set_of_char = set()
        for sequence in dataset:
            set_of_char = set_of_char.union(sequence)
        sorted_character_set = sorted(list(set_of_char))
        char_dictionary = {index : char for index,char in enumerate(self.special_symbol + sorted_character_set)}  #we want that special symbol have the first position
        return defaultdict(lambda : DEFAULT_UNKNOW_SYMBOL, char_dictionary)
    
    def get_padding_index(self) -> int:
        return self.char_index_dictionary[DEFAULT_PADDING_SYMBOL]
    
    def get_vocabolary_dimension(self) -> int:
        return len(self.char_index_dictionary)
    
    def get_max_length(self) -> int:
        return MAX_SEQUENCE_LEN + 2
        
def transform_data_to_token(seq2seq_dataset, transaltor_object = None):    #when dimension_of_dataset is equal to -1 we use the whole data
    input_sentence_dataset, output_sentence_dataset = get_clean_string_dataset(seq2seq_dataset)
    input_sequence_of_chars_dataset = sentence_as_a_list_of_chars(input_sentence_dataset)
    output_sequence_of_chars_dataset = sentence_as_a_list_of_chars(output_sentence_dataset)
    if(not transaltor_object):
        encoding_object = IndexTranslator(output_sequence_of_chars_dataset)
    else:
        encoding_object = transaltor_object
    return (create_regular_ML_dataset(input_sequence_of_chars_dataset,encoding_object), 
        create_regular_ML_dataset(output_sequence_of_chars_dataset,encoding_object),
        encoding_object)

def get_clean_string_dataset(dataset : list[str]):
    cleaned_dataset = remove_backslash(dataset)
    output_sentence_dataset = shorten_sequences_to_target_lenght(cleaned_dataset)
    input_sentence_dataset = make_input_dataset(output_sentence_dataset)
    return input_sentence_dataset, output_sentence_dataset


def load_dataset(file = TEST_INPUT_FILE) -> list[str]: 
    with open(file, encoding = "utf-8", errors= "ignore") as reference_to_dataset_file:
        return list(csv.reader(reference_to_dataset_file))  

def remove_backslash(dataset:list[str]):
    remove_backslash_func = substitute_symbol_func_factory(r"[\\]")
    dataset_without_backslash = list(map(remove_backslash_func,dataset))
    return list(map(lambda sentence : sentence.lower(),dataset_without_backslash))

def substitute_symbol_func_factory(regex_of_target:str,substitute = ""):
    def substitute_specific_symbol(target_string):
        return re.sub(pattern = regex_of_target, repl = substitute, string = target_string)    
    return substitute_specific_symbol

def sentence_as_a_list_of_chars(dataset:list[str]):
    substitute_space_function = substitute_symbol_func_factory(r"[ ]",ENCODING_DEFAULT_SEPARATOR_SYMBOL)
    dataset_whit_new_symbol_for_space = map(substitute_space_function, dataset)
    return [[char for char in sentence] for sentence in dataset_whit_new_symbol_for_space]                                                    

def embed_chars_in_spaces(target_string:str):
    substitution_func = lambda char : f" {char} "
    return substitute_simbols_in_string(target_string,substitution_func)

def substitute_simbols_in_string(target_string:str,substitution_func : Callable[[str],str]):
    list_chars_embedded = list(map(substitution_func, target_string))
    return "".join(list_chars_embedded,)


def shorten_sequences_to_target_lenght(long_sequence_dataset : list[str]):                                                    
    too_long = lambda seq : len(seq) > MAX_SEQUENCE_LEN
    long_sequences = list(filter(lambda seq: too_long(seq), long_sequence_dataset))
    ok_sequences = list(filter(lambda seq: not too_long(seq), long_sequence_dataset))
    del(long_sequence_dataset)
    ok_sequences += split_longer_sequences(long_sequences,too_long)
    return ok_sequences

def split_longer_sequences(long_sequences: list[str], too_long : Callable[[str],bool]):                                                      
    short_sequences = []
    while(long_sequences):
        splitted_list_of_list = list(map(split_in_middle_space,long_sequences)) 
        splitted_list = list(itertools.chain(*splitted_list_of_list))   #flatten the list of list
        long_sequences = list(filter(lambda seq : too_long(seq), splitted_list))
        short_sequences += list(filter(lambda seq : not too_long(seq), splitted_list ))
    return short_sequences

def split_in_middle_space(long_char_sequence : str):
    indexes_of_space = more_itertools.locate(long_char_sequence, lambda char : char == DATASET_WORD_SEPARATOR_SYMBOL) #return the list of index when a space is found
    worst_space_metrics = lambda index : abs(index - (len(long_char_sequence)/2))  #i want to split the sequence in the space that is more central
    try:
        best_index = min(indexes_of_space,key = worst_space_metrics)
    except:                  
        return []  #this appen when a single word is longer than the maximum lenght of sequence; in this case we ignore the world because we assume that is an error in the dataset
    return (long_char_sequence[:best_index],long_char_sequence[best_index+1:])


def make_input_dataset(list_of_sequences : list[str]):
        assert(len(max(list_of_sequences, key = len)) <= MAX_SEQUENCE_LEN)
        remove_separator_symbols_function = substitute_symbol_func_factory(f"{DATASET_WORD_SEPARATOR_SYMBOL}")
        return list(map(remove_separator_symbols_function,list_of_sequences))

def create_regular_ML_dataset(sequence_of_chars_dataset : list[list[str]], encoding_object : IndexTranslator):
    bos_and_eos_dataset = [[START_OF_SENTENCE_SYMBOL] + sentence + [END_OF_SENTENCE_SYMBOL] for sentence in sequence_of_chars_dataset]
    padded_dataset = create_padding_dataset(bos_and_eos_dataset)
    encoded_dataset = [encoding_object.encode_sequence(sequence) for sequence in padded_dataset]
    return encoded_dataset

def create_padding_dataset(sequence_of_chars_dataset : list[list[str]]) -> list[list[str]]:
    lenght_of_longest_sequence = len(max(sequence_of_chars_dataset, key=len))
    return list(map(
        lambda sequence_of_char : padding_to_distance(sequence_of_char, lenght_of_longest_sequence), 
        sequence_of_chars_dataset) )

def padding_to_distance(sequence_of_char :list[str], length_of_longest_sequence :int):
    distance_to_longest = length_of_longest_sequence - len(sequence_of_char)
    return sequence_of_char + [DEFAULT_PADDING_SYMBOL] * distance_to_longest

