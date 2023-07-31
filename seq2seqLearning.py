from seq2seqPreprocessing import transform_data_to_token,IndexTranslator
from modelArgumentManagement import ArgumentFromJson
from abc import abstractmethod
from sklearn.model_selection import train_test_split
import pathlib
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import one_hot, log_softmax
import trainTestSplitter

ENCODER_PARAMETER_DEFAULT_PATH = pathlib.Path("pytorch_encoder_parameters.json")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 6082023

class Seq2SeqDataset(Dataset):
    def __init__(self, input_senquence:list[list[str]], target_sequence: list[list[str]]):
        assert(len(input_senquence) == len(target_sequence))
        self.input_sequence = torch.tensor(input_senquence)
        self.target_sequence = torch.tensor(target_sequence)

    def __len__(self):
        return self.input_sequence.size()[0]
    
    def __getitem__(self, index:int) -> tuple[torch.tensor]:
        return self.input_sequence[index], self.target_sequence[index]

class ManageDataset():
    def __init__(self, seq2seq_train_dataset : list[str], seq2seq_test_dataset):
        self.train_input, self.train_output, self.translatorObject = transform_data_to_token(seq2seq_train_dataset)
        self.test_input, self.test_output, _  = transform_data_to_token(seq2seq_test_dataset, self.translatorObject)
    
    def get_translatorObject(self) -> IndexTranslator:
        return self.translatorObject
    
    def get_input(self) -> list[list[str]]:
        return self.train_input + self.test_input
    
    def get_output(self) -> list[list[str]]:
        return self.train_output + self.test_output
    
    def get_train_dataset(self) -> tuple[list[list[str], list[list[str]]]]:
        return self.train_input, self.train_output
    
    def get_test_dataset(self) -> tuple[list[list[str]], list[list[str]]] :
        return self.test_input, self.test_output
    
    def get_tensor_representation(self)  -> tuple[torch.Tensor]:
        return torch.tensor(self.get_input()), torch.tensor(self.get_output())


class InternalDataRepresentation():      #the idea behind this abstract class is to allow different encoding of data
    @abstractmethod
    def get_data_representation(self, source: torch.tensor):
        pass

    @abstractmethod 
    def get_data_representation_size(self) -> int:
        pass

    @abstractmethod 
    def get_data_original_size(self) -> int:
        pass

class OneHotEncoding(InternalDataRepresentation):
    def __init__(self, vocab_dictionary : IndexTranslator, num_head : int):
        self.dictionary_size = vocab_dictionary.get_vocabolary_dimension()
        self.max_len = vocab_dictionary.get_max_length()
        self.padding_symbol_index = vocab_dictionary.get_padding_index()
        self.normalize_size_to_num_head(num_head) #the embedding dimension must be divisble for num_head
       
        
    def get_data_representation_size(self) -> int:
        return self.dictionary_size + self.max_len
    
    def normalize_size_to_num_head(self, num_head):
        if self.get_data_original_size() % num_head != 0:
            to_next_multiple_value = (num_head - (self.dictionary_size + self.max_len) % num_head)
            self.max_len += to_next_multiple_value         
            
    def get_data_original_size(self) -> int:
        return self.dictionary_size
    
    def get_data_representation(self, source: torch.tensor) -> torch.tensor :
        source_embedding = self.encoding_data(source)
        position_embedding = self.encoding_position(source)
        source_with_zero, position_with_zero = self._zero_pad(source_embedding, position_embedding)
        
        return source_with_zero + position_with_zero
    
    def encoding_data(self, tensor_sequences : torch.tensor) -> torch.tensor:
        return one_hot(tensor_sequences, self.dictionary_size)
    
    def encoding_position(self, tensor_sequences : torch.tensor) -> torch.tensor:
        number_of_elements = tensor_sequences.size()[0]
        lenght_of_elements = tensor_sequences.size()[1]
        position_encoding = one_hot(torch.arange(lenght_of_elements),self.max_len)
        position_encoding_for_batch = position_encoding.expand(number_of_elements,lenght_of_elements,self.max_len)
        return position_encoding_for_batch
        
    
    def padding_index(self) -> int:
        return self.padding_symbol_index
    
    def _zero_pad(self, source_embedding :torch.tensor , position_embedding : torch.tensor):
        zero_tensor_source = torch.zeros(*source_embedding.size())
        zero_tensor_position = torch.zeros(*position_embedding.size())

        zeroed_source_embedding = torch.cat((source_embedding,zero_tensor_position), dim = 2)
        zeored_position_embedding = torch.cat((zero_tensor_source, position_embedding), dim = 2)
        return zeroed_source_embedding, zeored_position_embedding
        
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
        position_embedding = self.encoding_position(source)
        return self.dropout_model(source_embedding + position_embedding)

    def encoding_position(self, source: torch.tensor) -> torch.tensor:
        number_of_source_data, source_data_dimension = source.size()    #we work with a batch of data
        tensor_with_position = (
            torch.arange(0, source_data_dimension) #generate a tensor with all the index from 0 to the dimension of data
            .unsqueeze(1)         #create a tensor for all index
            .expand(source_data_dimension, number_of_source_data)  #expand each index tensor to the number of source data
            .transpose(0,1)     #transpose the matrix, now we have for each source data a tensor with all the index of position
            .to(DEVICE)
        )
        embedding_position_model = self._get_embedding_position_model(source_data_dimension)
        return embedding_position_model(tensor_with_position)
    
    def _get_embedding_position_model(self, source_data_dimension : int):        
        if( not self.embedding_position_models.get(source_data_dimension)):
            self.embedding_position_models[source_data_dimension] = torch.nn.Embedding(source_data_dimension,
                                                      self.embedding_data_dimension,
                                                      device = DEVICE)
        return self.embedding_position_models[source_data_dimension]

class EmbeddingRepresentetionConcat(EmbeddingRepresentation):
    
    def get_data_representation(self, source: torch.tensor) -> torch.tensor :
        source_embedding = self.data_embedding_model(source)
        position_embedding = self.encoding_position(source)
        source_with_zero, position_with_zero = self._zero_pad(source_embedding, position_embedding)
        return self.dropout_model(source_with_zero + position_with_zero)
    
    def _zero_pad(self, source_embedding :torch.tensor , position_embedding : torch.tensor):
        zero_tensor = torch.zeros(*source_embedding.size())
        zeroed_source_embedding = torch.cat((source_embedding,zero_tensor), dim = 2)
        zeored_position_embedding = torch.cat((zero_tensor, position_embedding), dim = 2)
        return zeroed_source_embedding, zeored_position_embedding
    
    def get_data_representation_size(self) -> int:
        return self.embedding_data_dimension * 2
    

class PytorchTransformerArguments(ArgumentFromJson):
    def __init__(self, encoder_parameters_file_path = ENCODER_PARAMETER_DEFAULT_PATH):
        super(PytorchTransformerArguments,self).__init__(encoder_parameters_file_path)

    def  get_arguments(self,number_of_input_features = 512 ) -> dict:
        self.parameters_dict['d_model'] = number_of_input_features
        self.parameters_dict['device' ]= DEVICE
        self.parameters_dict['batch_first']= True
        return self.parameters_dict
            
    
class Transformer(torch.nn.Module):
    def __init__(self, input_representation_object : InternalDataRepresentation, tranformers_argument : PytorchTransformerArguments):
        super(Transformer,self).__init__()
        self.internal_input_represantion = input_representation_object
        number_of_data_features = input_representation_object.get_data_representation_size()
        number_of_output_features = input_representation_object.get_data_original_size()
        arguments_of_transfomer_dict = tranformers_argument.get_arguments(number_of_data_features)
        self.transformer = torch.nn.Transformer(**arguments_of_transfomer_dict)
        feedforward_dimension = arguments_of_transfomer_dict.get("dim_feedforward")
        self.output_linear_layer = self.out = torch.nn.Sequential(
            torch.nn.Linear(number_of_data_features, feedforward_dimension),
            torch.nn.ReLU(), 
            torch.nn.Linear(feedforward_dimension, number_of_output_features))

    def forward(self, input_sequence : torch.tensor, target_sequence : torch.tensor):
            input_seq_representation = self.internal_input_represantion.get_data_representation(input_sequence)
            target_seq_representation = self.internal_input_represantion.get_data_representation(target_sequence)
            input_padding_mask = self.get_padding_mask(input_sequence)
            target_padding_mask = self.get_padding_mask(target_sequence)
            target_position_padding_mask = self.transformer.generate_square_subsequent_mask(target_sequence.size()[1]).to(torch.bool)
            transformer_output = self.transformer(input_seq_representation, 
                                     target_seq_representation,
                                     src_key_padding_mask = input_padding_mask,
                                     tgt_key_padding_mask = target_padding_mask,
                                     tgt_mask = target_position_padding_mask)
            return log_softmax(self.output_linear_layer(transformer_output), dim=-1)

    
    def get_padding_mask(self, tokenized_sequence : torch.tensor):
        padding_index = self.internal_input_represantion.padding_index()
        padding_data_mask = (tokenized_sequence == padding_index)  #this create a vector with the results of the comparison
        return padding_data_mask
    
 
class TrainTestTransformer():
    def __init__(self, model : Transformer, 
                  batch_size :float,
                   criterion) -> None:
        self.batch_size = batch_size
        self.model = model.to(DEVICE)
        self.criterion = criterion
    
    def get_dataset_iterator_object(self, dataset : tuple[torch.tensor, torch.tensor]) -> DataLoader:
        training_dataset = Seq2SeqDataset(*dataset)
        return DataLoader(training_dataset,self.batch_size)
    
    def move_batch_to_device(self, batch_input_data : torch.tensor , batch_target_data : torch.tensor) -> tuple[torch.tensor, torch.tensor]:
        model_input_data = batch_input_data.to(DEVICE)
        target_data = batch_target_data.to(DEVICE)
        return model_input_data, target_data
  
    def feed_model(self, model_input_data :torch.tensor , target_data : torch.tensor) -> torch.tensor:
        target_for_model = target_data[:,:-1] #remove the token that signal the end of sentence in the train set
        return self.model(model_input_data, target_for_model)  
    
    def reshape_for_loss_function(self, output_of_model: torch.tensor, 
                                  target_sequence : torch.tensor) -> tuple[torch.tensor,torch.tensor]: 
        output_for_loss = output_of_model.reshape(-1, output_of_model.shape[2])    
        target_for_loss = target_sequence[:,1:].reshape(-1)# remove the begin of sentence token in the evaluation step
        return output_for_loss, target_for_loss
    
    
    