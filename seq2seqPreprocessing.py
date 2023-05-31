import csv
from pathlib import Path
import regex as re
from typing import Callable
import more_itertools
import itertools

DATASET_DIRCECTORY = Path("dataset")
TEST_INPUT_FILE = Path(DATASET_DIRCECTORY,"train_output.csv") 
ENCODING_DEFAULT_SEPARATOR_SYMBOL = "#"
DATASET_WORD_SEPARATOR_SYMBOL = ' '
MAX_SEQUENCE_LEN = 120
END_OF_SENTENCE_SYMBOL = "EOS"  
START_OF_SENTENCE_SYMBOL = "SOS"

class IndexTranslator():

    def __init__(self,index_char_dictionary: dict[int:str]) -> None:
        self.index_char_dictionary = index_char_dictionary
        self.char_index_dictionary = {char:index for index,char in index_char_dictionary.items()}
    
    def encode_sequence(self, sentence_as_a_sequence :list[str]):
        return [self.char_index_dictionary[char] for char in sentence_as_a_sequence]
    
    def sequence_from_encode(self, encoded_sequence :list[int]):
        return [self.index_char_dictionary[index] for index in encoded_sequence]



def load_dataset(): 
    """I use only the output file because i want to fine-tune a model that learn how to put spaces in text,
    #so the input must be a a text without spaces and the outut the text with the spaces in the correct
    #position. To build this type of dataset i need a dataset with the spaces in the correct position, 
    #so i use the output file."""
    reference_to_dataset_file = open(TEST_INPUT_FILE, encoding = "utf-8", errors= "ignore")
    return list(csv.reader(reference_to_dataset_file))  
    

def clean_dataset(dataset:list[str]):
    dataset_without_schema = dataset[1:]
    remove_id_field_func =  lambda list_id_str : list_id_str[1] #the first row of the dataset is the description of the schema 
                                                                #(id,string), every row after the first is a list with an 
                                                                # id in first position and a string in the second position
    dataset_with_only_string = list(map(remove_id_field_func, dataset_without_schema))
    remove_backslash_func = substitute_symbol_func_factory(r"[\\]")
    dataset_without_backslash = list(map(remove_backslash_func,dataset_with_only_string))
    return dataset_without_backslash

def substitute_symbol_func_factory(regex_of_target:str,substitute = ""):
    def substitute_specific_symbol(target_string):
        return re.sub(pattern = regex_of_target, repl = substitute, string = target_string)    
    return substitute_specific_symbol

def sentence_as_a_list_of_chars(dataset:list[str]):
    substitute_space_function = substitute_symbol_func_factory(r"[ ]",ENCODING_DEFAULT_SEPARATOR_SYMBOL)
    dataset_whit_new_symbol_for_space = map(substitute_space_function, dataset)
    return [[START_OF_SENTENCE_SYMBOL]+[char for char in sentence]+[END_OF_SENTENCE_SYMBOL] for sentence in dataset_whit_new_symbol_for_space]                                                    

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

def split_longer_sequences(long_sequences:list[str], too_long :Callable[[str],bool]):                                                      
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

def create_char_dictionary(dataset : list [list[str]]):
    set_of_char = set([START_OF_SENTENCE_SYMBOL,END_OF_SENTENCE_SYMBOL])
    for sequence in dataset:
        set_of_char = set_of_char.union(sequence)
    char_dictionary = {index : char for index,char in enumerate(set_of_char)}
    return char_dictionary


def make_input_dataset(list_of_sequences : list[list[str]]):
        assert(len(max(list_of_sequences, key = len)) <= MAX_SEQUENCE_LEN)
        remove_separator_symbols_function = substitute_symbol_func_factory(f"{ENCODING_DEFAULT_SEPARATOR_SYMBOL}")
        return list(map(remove_separator_symbols_function,list_of_sequences))

def create_regular_ML_dataset(sequence_of_chars_dataset : list[list[str]], encoding_object):
    encoded_dataset = [encoding_object.encode_sequence(sequence) for sequence in sequence_of_chars_dataset]
    not_used_index = len(encoding_object.index_char_dictionary)
    lenght_of_longest_sequence = len(max(encoded_dataset,key=len))
    return list(map(lambda sequence_of_char : padding_to_distance(sequence_of_char,lenght_of_longest_sequence,not_used_index), encoded_dataset))


def padding_to_distance(sequence_of_char :list[str], length_of_longest_sequence :int, padding_symbol : int):
    distance_to_longest = length_of_longest_sequence - len(sequence_of_char)
    return sequence_of_char + [padding_symbol for _ in range(distance_to_longest)]

def transform_data(number_of_sample = 20):
    dataset_sample = load_dataset()[:number_of_sample]
    cleaned_dataset_sample = clean_dataset(dataset_sample)
    output_sentence_dataset = shorten_sequences_to_target_lenght(cleaned_dataset_sample)
    input_sentence_dataset = make_input_dataset(output_sentence_dataset)   #the input dataset is equal to the output without the spaces
    input_sequence_of_chars_dataset = sentence_as_a_list_of_chars(input_sentence_dataset)
    output_sequence_of_chars_dataset = sentence_as_a_list_of_chars(output_sentence_dataset)
    char_dictionary = create_char_dictionary(output_sequence_of_chars_dataset)
    encoding_object = IndexTranslator(char_dictionary)

    return create_regular_ML_dataset(input_sequence_of_chars_dataset,encoding_object), create_regular_ML_dataset(output_sequence_of_chars_dataset,encoding_object)

