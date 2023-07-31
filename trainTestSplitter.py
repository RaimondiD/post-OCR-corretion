from sklearn.model_selection import train_test_split
import pandas as pd

INPUT_DATASET_PATH = "dataset/train_input.csv"
OUTPUT_DATASET_PATH = "dataset/train_output.csv"
DATA_LABEL_INPUT_DATASET = "original"
DATA_LABEL_OUTPUT_DATASET = "solution"
random_seed = 15062023

class TrainTestSplitter():
    def __init__(self,dim_of_sample = -1):
        input_dataset = TrainTestSplitter._load_dataset(INPUT_DATASET_PATH).head(dim_of_sample)
        output_datase = TrainTestSplitter._load_dataset(OUTPUT_DATASET_PATH).head(dim_of_sample)
        train_and_test_pandasDF = train_test_split(
            input_dataset, 
            output_datase,
            random_state=random_seed,
            train_size=0.7) 
        self.train_input_dataset, self.test_input_dataset, self.train_output_dataset, self.test_output_dataset = list(map(
            lambda pandasDataset : pandasDataset.to_dict('list'), 
            train_and_test_pandasDF)) 
    
    def _load_dataset(dataset_path) -> pd.DataFrame:
        dictionary_for_row = pd.read_csv(dataset_path)
        return dictionary_for_row
    
    def get_seq2seq_train_dataset(self)-> list[str]:
        return self.train_output_dataset[DATA_LABEL_OUTPUT_DATASET]
    
    def get_seq2seq_test_dataset(self) -> list[str]:
        return self.test_output_dataset[DATA_LABEL_OUTPUT_DATASET]

    def get_BERT_train_data(self) -> tuple[dict[str,list[str]], dict[str, list[str]]]:
        return self.train_input_dataset,  self.train_output_dataset
    
    def get_BERT_test_data(self) -> tuple[dict[str,list[str]], dict[str, list[str]]]:
        return self.test_input_dataset, self.test_output_dataset
    
    def get_dictionary_data(self) -> list[str]:
        return self.train_output_dataset[DATA_LABEL_OUTPUT_DATASET]
    
    def get_test_string_datasets(self) -> tuple[list[str],list[str]]:
        return self.test_input_dataset[DATA_LABEL_INPUT_DATASET], self.test_output_dataset[DATA_LABEL_OUTPUT_DATASET]