from seq2seqPreprocessing import transform_data_to_token,IndexTranslator
from abc import abstractmethod
from sklearn.model_selection import train_test_split
import torch
import json

DEFAULT_ENCODER_PARAMETER_FILE = "pytorch_encoder_parameters.json"
DEVICE = 'cpu'


def get_dataset():
    input, output,_ = transform_data_to_token()
    print(f"input padded len = {len(input[0])}, output padded len : {len(output[0])}")

    train_input, test_input, train_output, test_output = train_test_split(input,output)
    train_input_tensor = torch.tensor(train_input, dtype = torch.int )
    train_output_tensor = torch.tensor(train_output, dtype = torch.int)
    test_input_tensor = torch.tensor(test_input, dtype = torch.int)
    test_output_tensor = torch.tensor(test_output, dtype = torch.int)
    return train_input_tensor, train_output_tensor, test_input_tensor, test_output_tensor


class InternalDataRepresentation():
    @abstractmethod
    def get_data_representation(self, source:torch.tensor):
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
                 dropout_embedding_percentual : float):
        
        self.padding_symbol_index = vocab_dictionary.get_padding_index()
        self.dictionary_size = vocab_dictionary.get_vocabolary_dimension()
        self.embedding_data_dimension = embedding_data_dimension

        self.dropout_model = torch.nn.Dropout(dropout_embedding_percentual)
        self.data_embedding_model = torch.nn.Embedding(self.dictionary_size, 
                                                       embedding_data_dimension, 
                                                       self.padding_symbol_index, 
                                                       device= DEVICE)
        self.embedding_position_models = {}   #the position embedded depend on the dimension of the data passed to it, in this
                                             #case input and output data can have different dimension, so we use a dict to save
                                             #the positional embedding associated with every position, we manage it with the
                                             #method _get_embedded_position_model
    
    def get_data_representation_size(self) -> int:
        return self.embedding_data_dimension
    
    def padding_index(self) -> int:
        return self.padding_symbol_index


    def get_data_original_size(self) -> int:
        return self.dictionary_size

    def get_data_representation(self, source: torch.tensor):
        source_embedding = self.data_embedding_model(source)
        position_embedding = self.__encoding_position(source)
        return self.dropout_model(source_embedding + position_embedding)

    def __encoding_position(self, source: torch.tensor) -> torch.tensor:
        number_of_source_data, source_data_dimension = source.size()    #we work with a batch of data

        tensor_with_position = (
            torch.arange(0, source_data_dimension) #generate a tensor with all the index from 0 to the dimension of data
            .unsqueeze(1)         #create a tensor for all index
            .expand(source_data_dimension, number_of_source_data)  #expand each index tensor to the number of source data
            .transpose(0,1)     #transpose the matrix, now we have for each source data a tensor with all the index of position
            .to(DEVICE)
        )
        embedding_position_model = self.__get_embedding_position_model(source_data_dimension)
        return embedding_position_model(tensor_with_position)
    
    def __get_embedding_position_model(self, source_data_dimension : int):        
        if( not self.embedding_position_models.get(source_data_dimension)):
            self.embedding_position_models[source_data_dimension] = torch.nn.Embedding(source_data_dimension,
                                                      self.embedding_data_dimension,
                                                      device = DEVICE)
        return self.embedding_position_models[source_data_dimension]

class PytorchTransformerArguments():
    def __init__(self, encoder_parameters_file_path =DEFAULT_ENCODER_PARAMETER_FILE):
        with open(encoder_parameters_file_path,"r") as arguments_file:
            self.parameters_dict = json.load(arguments_file) 
    
    def  get_arguments(self,number_of_input_features = 512 ) -> dict:
        self.parameters_dict['d_model'] = number_of_input_features
        self.parameters_dict['device']=DEVICE
        self.parameters_dict['batch_first']= True
        return self.parameters_dict



class Transformer(torch.nn.Module):
    def __init__(self, input_representation_object : InternalDataRepresentation, tranformers_argument : PytorchTransformerArguments):
        super(Transformer,self).__init__()
        self.internal_input_represantion = input_representation_object
        number_of_data_features = input_representation_object.get_data_representation_size()
        number_of_output_features = input_representation_object.get_data_original_size()
        self.transformer = torch.nn.Transformer(**tranformers_argument.
                                                 get_arguments(number_of_data_features))
        self.output_linear_layer = torch.nn.Linear(number_of_data_features,number_of_output_features)

    def forward(self, input_sequence : torch.tensor, target_sequence : torch.tensor):
            input_seq_representation = self.internal_input_represantion.get_data_representation(input_sequence)
            target_seq_representation = self.internal_input_represantion.get_data_representation(target_sequence)
            input_padding_mask = self.get_padding_mask(input_sequence)
            target_padding_mask = self.get_padding_mask(target_sequence)
            target_position_padding_mask = self.transformer.generate_square_subsequent_mask(target_sequence.size()[1])
            transformer_output = self.transformer.forward(input_seq_representation, 
                                     target_seq_representation, 
                                     src_key_padding_mask = input_padding_mask,
                                     tgt_key_padding_mask = target_padding_mask,
                                     tgt_mask = target_position_padding_mask)
            return self.output_linear_layer(transformer_output)


    def get_padding_mask(self, tokenized_sequence : torch.tensor):
        padding_index = self.internal_input_represantion.padding_index()
        padding_data_mask = tokenized_sequence == padding_index  #this create a vector with the results of the comparison
        return padding_data_mask
            
    
    
class TrainingTransformer():
    def __init__(self) -> None:
        pass

    def train_model(train_input_dataset : torch.tensor, train_output_dataset : torch.tensor):
        pass

    def evaluate_model(test_input_dataset, test_output_dataset : torch.tensor):
        pass