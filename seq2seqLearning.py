from seq2seqPreprocessing import transform_data_to_token,IndexTranslator
from abc import abstractmethod
from sklearn.model_selection import train_test_split
import torch
import json

DEFAULT_ENCODER_PARAMETER_FILE = "pytorch_encoder_parameters.json"
DEVICE = 'cpu'
def get_dataset():
    input,output = transform_data_to_token()
    train_input, train_output, test_input, test_output = train_test_split(input,output)
    train_input_tensor = torch.tensor(train_input, dtype = torch.int )
    train_output_tensor = torch.tensor(train_output, dtype = torch.int)
    test_input_tensor = torch.tensor(test_input, dtype = torch.int)
    test_output_tensor = torch.tensor(test_output, dtype = torch.int)
    return train_input_tensor, train_output_tensor, test_input_tensor, test_output_tensor

class InternalDataRepresentation():
    @abstractmethod
    def get_data_representation(self,source:torch.tensor):
        pass
    @abstractmethod 
    def get_data_representation_size(self) -> int:
        pass

    @abstractmethod 
    def get_data_original_size(self)->int:
        pass


class EmbeddingRepresentation(InternalDataRepresentation):
    def __init__(self,vocab_dictionary : IndexTranslator, 
                 embedding_data_dimension: int, 
                 embedding_position_dimension:int, 
                 dropout_embedding_percentual : float):
        
        self.padding_symbol_index = vocab_dictionary.get_padding_index()
        self.dictionary_size = vocab_dictionary.get_vocabolary_dimension()
        self.embedding_data_dimension = embedding_data_dimension
        self.embedding_position_dimension = embedding_position_dimension
        self.dropout_model = torch.nn.Dropout(dropout_embedding_percentual)
        self.data_embedding_model = torch.nn.Embedding(self.dictionary_size, 
                                                       embedding_data_dimension, 
                                                       self.padding_symbol_index, 
                                                       device= DEVICE)
    
    def get_data_representation_size(self) -> int:
        return self.embedding_data_dimension + self.embedding_position_dimension
    
    def get_data_original_size(self) -> int:
        return self.dictionary_size
    
    def get_data_representation(self, source: torch.tensor):
        source_embedding = self.data_embedding_model(source)

    def __encoding_position(self, source: torch.tensor) -> torch.tensor:
        source_data_dimension, number_of_source_data = source.size()   
        tensor_with_position = (
            torch.arange(0, source_data_dimension) #generate a tensor with all the index from 0 to the dimension of data
            .unsqueeze(1)         #create a tensor for all index
            .expand(source_data_dimension, number_of_source_data)  #expand each index tensor to the number of source data
            .transpose(0,1)     #transpose the matrix, now we have for each source data a tensor with all the index of position
            .to(DEVICE)
        )
        embedding_position_model = torch.nn.Embedding(source_data_dimension,
                                                      self.embedding_position_dimension,
                                                      device = DEVICE)
        return embedding_position_model(tensor_with_position)



class PytorchTransformerArguments():
    def __init__(self, encoder_parameters_file_path =DEFAULT_ENCODER_PARAMETER_FILE):
        with open(encoder_parameters_file_path,"r") as arguments_file:
            self.parameters_dict = json.load(arguments_file) 
    
    def  get_arguments(self,number_of_input_features = 512 ) -> dict:
        self.parameters_dict['d_model'] = number_of_input_features
        return self.parameters_dict




class Transformer(torch.nn.Module):
    def __init__(self, input_representation_object : InternalDataRepresentation, tranformers_argument : PytorchTransformerArguments):
        super(Transformer,self).__init__()
        self.internal_input_represantion = input_representation_object
        number_of_data_features = input_representation_object.get_data_representation_size()
        number_of_output_features = input_representation_object.get_data_original_size()
        self.transformers = torch.nn.Transformer(**tranformers_argument.
                                                 get_arguments(number_of_data_features), device=DEVICE)
        self.output_linear_layer = torch.nn.Linear(number_of_data_features,number_of_output_features)


def test():  
    dummy_encode = InternalDataRepresentation()
    a = PytorchTransformerArguments()
    dummyTransformer = Transformer(dummy_encode,a)

test()
