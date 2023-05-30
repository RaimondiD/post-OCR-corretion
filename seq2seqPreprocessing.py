import csv
from pathlib import Path
import regex as re
from typing import Callable
import more_itertools
import itertools
import transformers
from torch.utils.data import DataLoader, TensorDataset
import torch

DATASET_DIRCECTORY = Path("dataset")
TEST_INPUT_FILE = Path(DATASET_DIRCECTORY,"train_output.csv") 
DEFAULT_SEPARATOR_SYMBOL = "#"
MAX_SEQUENCE_LEN = 300
END_OF_SENTENCE_SYMBOL = "€"  
SUBSTITUTE_TO_EOS_SYMBOL = "$"



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
    substitute_EOS_symbol = substitute_symbol_func_factory(f"[{END_OF_SENTENCE_SYMBOL}]",f"{SUBSTITUTE_TO_EOS_SYMBOL}")
    dataset_without_EOS_simbol = list(map(substitute_EOS_symbol,dataset_without_backslash))
    return dataset_without_EOS_simbol

def substitute_symbol_func_factory(regex_of_target:str,substitute = ""):
    def remove_specific_symbol(target_string):
        return re.sub(pattern = regex_of_target, repl = substitute, string = target_string)    
    return remove_specific_symbol


def separate_chars_with_spaces(dataset:list[str]):
    dataset_whit_sub_spaces = map(change_separator_symbol,dataset)
    return list(map(embed_chars_in_spaces,dataset_whit_sub_spaces))

def change_separator_symbol(target_string:str):
    substitution_func = lambda char : char if char != " " else DEFAULT_SEPARATOR_SYMBOL
    return substitute_simbols_in_string(target_string,substitution_func)

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
        short_sequences = list(filter(lambda seq : not too_long(seq), splitted_list ))
    return short_sequences

def split_in_middle_space(long_char_sequence : str):
    indexes_of_space = more_itertools.locate(long_char_sequence, lambda char : char == DEFAULT_SEPARATOR_SYMBOL) #return the list of index when a space is found
    worst_space_metrics = lambda index : abs(index - (len(long_char_sequence)/2))  #i want to split the sequence in the space that is more central
    try:
        best_index = min(indexes_of_space,key = worst_space_metrics)
    except:
        return []
    return (long_char_sequence[:best_index],long_char_sequence[best_index+1:])


def make_output_dataset(list_of_sequences : list[str]):
    assert(len(max(list_of_sequences, key = len)) <= MAX_SEQUENCE_LEN)
    return list(map(lambda seq : seq + END_OF_SENTENCE_SYMBOL,list_of_sequences))

def make_input_dataset(list_of_sequences : list[str]):
        assert(len(max(list_of_sequences, key = len)) <= MAX_SEQUENCE_LEN)
        remove_separator_symbols_function = substitute_symbol_func_factory(f" {DEFAULT_SEPARATOR_SYMBOL} ")
        return list(map(remove_separator_symbols_function,list_of_sequences))

def encode_for_T5(input_dataset,output_dataset):
    tokenizer_t5 = transformers.T5Tokenizer.from_pretrained('t5-small')
    dict_of_tokenized_dataset = tokenizer_t5(input_dataset,text_target = output_dataset, 
                                             max_length = MAX_SEQUENCE_LEN, 
                                            truncation = True, padding = True, 
                                            return_tensors = "pt")
    return dict_of_tokenized_dataset['input_ids'], dict_of_tokenized_dataset['attention_mask'], dict_of_tokenized_dataset['labels']
    
    

class Seq2SeqDataset(torch.utils.data.Dataset):
    def __init__(self, input_texts, target_texts, tokenizer):
        self.input_texts = input_texts
        self.target_texts = target_texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        input_text = self.input_texts[idx]
        target_text = self.target_texts[idx]
        tokenized = self.tokenizer(input_text,text_target = target_text, 
                                             max_length = MAX_SEQUENCE_LEN, 
                                            truncation = True, padding = True, 
                                            return_tensors = "pt")
        print(tokenized["input_ids"].size(),tokenized[0]['labels'].size)
        return tokenized["input_ids"], tokenized['attention_mask'], tokenized["labels"]

from transformers import DataCollatorForSeq2Seq

class CustomDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features):
        input_ids = [f[0] for f in features]
        labels = [f[1] for f in features]

        batch = self.tokenizer.pad({"input_ids": input_ids, "labels": labels}, return_tensors="pt")
        return batch


dataset_sample = load_dataset()[:20]
cleaned_dataset_sample = clean_dataset(dataset_sample)
sequences_of_char_sample = separate_chars_with_spaces(cleaned_dataset_sample)
good_length_sequences_of_char_sample = shorten_sequences_to_target_lenght(sequences_of_char_sample)
output_dataset_sample = make_output_dataset(good_length_sequences_of_char_sample)
input_dataset_sample = make_input_dataset(good_length_sequences_of_char_sample)
tokenizer_t5 = transformers.T5Tokenizer.from_pretrained('t5-small')
batch_size = 8
dataset = Seq2SeqDataset(input_dataset_sample,output_dataset_sample, tokenizer_t5)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
model = transformers.T5ForConditionalGeneration.from_pretrained('t5-small')
data_collator = CustomDataCollator(tokenizer_t5)
training_args = transformers.TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    logging_dir='./logs',
)

trainer = transformers.Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,  
    data_collator= data_collator
)
trainer.train()