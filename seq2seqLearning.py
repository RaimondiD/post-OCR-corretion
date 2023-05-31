from seq2seqPreprocessing import transform_data
from sklearn.model_selection import train_test_split
import torch

def get_dataset():
    input,output = transform_data()
    train_input, train_output, test_input, test_output = train_test_split(input,output)
    train_input_tensor = torch.tensor(train_input, dtype = torch.int )
    train_output_tensor = torch.tensor(train_output, dtype = torch.int)
    test_input_tensor = torch.tensor(test_input, dtype = torch.int)
    test_output_tensor = torch.tensor(test_output, dtype = torch.int)
    return train_input_tensor, train_output_tensor, test_input_tensor, test_output_tensor

