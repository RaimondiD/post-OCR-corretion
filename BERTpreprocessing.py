import seq2seqPreprocessing
from datasets import Dataset, Sequence, ClassLabel
from trainTestSplitter import TrainTestSplitter 
from transformers import AutoTokenizer, DataCollatorForTokenClassification

DATASET_SEPARATOR_SYMBOL = " "
DEFAULT_DATA_COLUMN_NAME = "token"
DEFAULT_PADDING_VALUE = -100

class DatasetPreprocessor():
    def __init__(self, dataset_object: TrainTestSplitter):
        self.train_data = dataset_object.get_BERT_train_data()
        self.test_data = dataset_object.get_BERT_test_data()
        self.dataset_dict = self._get_train_and_test_dataset()
        
    def _get_train_and_test_dataset(self):
        train_dataset = DatasetPreprocessor._get_dataset(self.train_data)
        test_dataset = DatasetPreprocessor._get_dataset(self.test_data)
        dict_dataset = {"train" : train_dataset, "test" : test_dataset}
        return dict_dataset
    
    def _get_dataset(data_tuple) -> Dataset:
        input_data, output_data = data_tuple         
        input_clean_dataset = DatasetPreprocessor._clean_dataset(input_data)
        output_clean_dataset = DatasetPreprocessor._clean_dataset(output_data)
        input_data = input_clean_dataset[DEFAULT_DATA_COLUMN_NAME]
        output_data = output_clean_dataset[DEFAULT_DATA_COLUMN_NAME]
        column_with_labels = DatasetPreprocessor._get_dataset_labels(input_data, output_data)
        NAME_OF_LABELS_COLUMN = "word_labels"
        return input_clean_dataset.add_column(NAME_OF_LABELS_COLUMN, 
                                    column_with_labels) \
                                    .cast_column(NAME_OF_LABELS_COLUMN,
                                                Sequence(ClassLabel(names=[0,1])))

    def _clean_dataset(input_dataset) -> Dataset :
        input_dataset = Dataset.from_dict(input_dataset)                
        clean_dataset_function = DatasetPreprocessor._get_map_function_from_function_on_data(seq2seqPreprocessing.remove_backslash)
        input_clean_dataset = input_dataset.map(clean_dataset_function, batched=True) 
        split_data_function = DatasetPreprocessor._get_map_function_from_function_on_data(DatasetPreprocessor._get_data_splitted)
        input_splitted_dataset = input_clean_dataset.map(split_data_function, batched = True)
        data_column_name = DatasetPreprocessor._get_data_column_name(input_dataset)
        return input_splitted_dataset.rename_column(data_column_name, DEFAULT_DATA_COLUMN_NAME)
                        
    def _get_map_function_from_function_on_data(function_to_map_data):
        return lambda dataset : DatasetPreprocessor._apply_function_on_data(dataset, function_to_map_data)

    def _apply_function_on_data(data_batch : dict, function_on_data : callable) -> Dataset :
        dataset_data_name = list(data_batch)[1]       #data_batch is a dict, so i extract the second key that is the name
                                                      #data column
        elaborate_data = function_on_data(data_batch[dataset_data_name])
        return {dataset_data_name : elaborate_data}
  
    def _get_data_column_name(dataset : Dataset) -> str:
        return dataset.column_names[1]
    
   
    def _get_data_splitted(dataset):
            return list(map(lambda string : string.split(DATASET_SEPARATOR_SYMBOL), dataset))

    def _get_dataset_labels(input_data : list[list[str]], output_data : list[list[str]]):
        lable_dataset = []
        for input_sequence, output_sequence in zip(input_data, output_data):
            token_labels = DatasetPreprocessor._get_token_label(input_sequence, output_sequence)
            lable_dataset.append(token_labels)
        return lable_dataset

    def _get_token_label(input_sequence : list[str], output_sequence : list[str])-> list[str]:
        token_lable = []
        len_difference = abs(len(input_sequence) - len(output_sequence))
        for i, word in enumerate(input_sequence):
            #The idea is to check if the word in the input is also in the output in a similar position.
            #To do so i use a window with a size proportional to the differnce of the lenght of the two sequences
            left_windows_index = max(0, i - len_difference - 1)
            right_windows_index = min(len(output_sequence), i + len_difference +1)
            if word in output_sequence[left_windows_index : right_windows_index]: 
                token_lable.append(1) #1 is the label assigned at the right word
            else:
                token_lable.append(0) #0 is the label assigned at the wrong word
                
        return token_lable
            
    def get_dataset_dict(self) -> dict[str, Dataset]:
        return self.dataset_dict
  
    def get_train_dataset(self) -> Dataset:
        return self.dataset_dict['train']
    
    def get_test_dataset(self) -> Dataset :
        return self.dataset_dict['test']

class DatasetTokenizer():
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.data_collator = DataCollatorForTokenClassification(self.tokenizer)
    
    def tokenize_dataset(self, dataset : Dataset) -> Dataset:
        tokenized_dataset = dataset.map(lambda dataset : self._tokenize_function(dataset), batched = True)
        return tokenized_dataset        
        
    def _tokenize_function(self, dataset):
        tokenized_batch = self.tokenizer(dataset[DEFAULT_DATA_COLUMN_NAME], truncation= True, is_split_into_words= True)
        alligned_label = self._allign_ids_to_feature(tokenized_batch, dataset)
        tokenized_batch['labels'] = alligned_label
        return tokenized_batch
    
    def _allign_ids_to_feature(self, tokenized_batch, dataset : Dataset) -> list[list[int]]:
        labels = []
        for i, label in enumerate(dataset['word_labels']):
            word_ids = tokenized_batch.word_ids(batch_index=i)
            label_ids = []        
            previous_word_idx = None
            for word_idx in word_ids:  # Set to -100 the special token. We want also to set to -100 all the tokens assigned
                                        # to a word except for the first.                
                if word_idx is None or previous_word_idx == word_idx :
                    label_ids.append(DEFAULT_PADDING_VALUE)
                else:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                previous_word_idx = word_idx
            labels.append(label_ids)
        return labels 
    
    def get_data_collator(self):
        return self.data_collator 
    