from seq2seqPreprocessing import transform_data_to_token,IndexTranslator
from abc import abstractmethod
from sklearn.model_selection import train_test_split
import os
import pathlib
import torch
import json
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import one_hot
from torch.utils.tensorboard import SummaryWriter

ENCODER_PARAMETER_DEFAULT_PATH = pathlib.Path("pytorch_encoder_parameters.json")
TRAINING_PARAMETER_DEFAULT_PATH = pathlib.Path("training_object_parameters.json")
SAVE_MODEL_PATH = pathlib.Path("models/checkpoint")
TENSORBOARD_RESULT = pathlib.Path("models/tensorboard/seq2seq_tran_loss")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    def __init__(self, last_index=-1):
        input_sequence_data, output_sequence_data, self.translatorObject = transform_data_to_token(last_index)
        self.train_input, self.test_input, self.train_output, self.test_output = \
            train_test_split(input_sequence_data,output_sequence_data)

    def get_translatorObject(self) -> IndexTranslator:
        return self.translatorObject
    
    def get_input(self) -> list[list[str]]:
        return self.train_input + self.test_input
    
    def get_output(self) -> list[list[str]]:
        return self.train_output + self.test_output
    
    def get_train_dataset(self) -> list[list[str]]:
        return self.train_input, self.train_output
    
    def get_test_dataset(self) -> list[list[str]] :
        return self.train_input, self.train_output
    
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
        source_with_zero, position_with_zero = self.__zero_pad(source_embedding, position_embedding)
        
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
    
    def __zero_pad(self, source_embedding :torch.tensor , position_embedding : torch.tensor):
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
        embedding_position_model = self.__get_embedding_position_model(source_data_dimension)
        return embedding_position_model(tensor_with_position)
    
    def __get_embedding_position_model(self, source_data_dimension : int):        
        if( not self.embedding_position_models.get(source_data_dimension)):
            self.embedding_position_models[source_data_dimension] = torch.nn.Embedding(source_data_dimension,
                                                      self.embedding_data_dimension,
                                                      device = DEVICE)
        return self.embedding_position_models[source_data_dimension]

class EmbeddingRepresentetionConcat(EmbeddingRepresentation):
    
    def get_data_representation(self, source: torch.tensor) -> torch.tensor :
        source_embedding = self.data_embedding_model(source)
        position_embedding = self.encoding_position(source)
        source_with_zero, position_with_zero = self.__zero_pad(source_embedding, position_embedding)
        return self.dropout_model(source_with_zero + position_with_zero)
    
    def __zero_pad(self, source_embedding :torch.tensor , position_embedding : torch.tensor):
        zero_tensor = torch.zeros(*source_embedding.size())
        zeroed_source_embedding = torch.cat((source_embedding,zero_tensor), dim = 2)
        zeored_position_embedding = torch.cat((zero_tensor, position_embedding), dim = 2)
        return zeroed_source_embedding, zeored_position_embedding
    
    def get_data_representation_size(self) -> int:
        return self.embedding_data_dimension * 2
    
class ArgumentFromJson : 
    def __init__(self, parameters_file_path : str):
        with open(parameters_file_path,"r") as arguments_file:
            self.parameters_dict = json.load(arguments_file) 

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
        self.transformer = torch.nn.Transformer(**tranformers_argument.
                                                 get_arguments(number_of_data_features))
        self.output_linear_layer = torch.nn.Linear(number_of_data_features,number_of_output_features)

    def forward(self, input_sequence : torch.tensor, target_sequence : torch.tensor):
            input_seq_representation = self.internal_input_represantion.get_data_representation(input_sequence)
            target_seq_representation = self.internal_input_represantion.get_data_representation(target_sequence)
            input_padding_mask = self.get_padding_mask(input_sequence)
            target_padding_mask = self.get_padding_mask(target_sequence)
            target_position_padding_mask = self.transformer.generate_square_subsequent_mask(target_sequence.size()[1]).to(torch.boo)
            transformer_output = self.transformer(input_seq_representation, 
                                     target_seq_representation,
                                     src_key_padding_mask = input_padding_mask,
                                     tgt_key_padding_mask = target_padding_mask,
                                     tgt_mask = target_position_padding_mask)
            return self.output_linear_layer(transformer_output)
    
    def get_padding_mask(self, tokenized_sequence : torch.tensor):
        padding_index = self.internal_input_represantion.padding_index()
        padding_data_mask = (tokenized_sequence == padding_index)  #this create a vector with the results of the comparison
        return padding_data_mask
    

class TrainingTransformerHyperparameters(ArgumentFromJson):
    def __init__(self, training_transformer_hyperparameters_path = TRAINING_PARAMETER_DEFAULT_PATH):
        super(TrainingTransformerHyperparameters,self).__init__(training_transformer_hyperparameters_path)
    
    def get_arguments(self, model : Transformer) -> dict :
        self.parameters_dict['model'] = model
        return self.parameters_dict

 
class TrainingAndTestTransformer():
    def __init__(self, model : Transformer, 
                 learning_rate : float,
                  batch_size :float,
                   num_epochs : int ) -> None:
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.model = model.to(DEVICE)
        self.writer = SummaryWriter(TENSORBOARD_RESULT)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = learning_rate)


    def __get_train_iterator_object(self, preprocessed_data_manager) -> DataLoader:
        training_dataset = Seq2SeqDataset(*preprocessed_data_manager.get_train_dataset())
        return DataLoader(training_dataset,self.batch_size)

    def __get_test_iterator_object(self, preprocessed_data_manager) -> DataLoader:
        testing_dataset = Seq2SeqDataset(*preprocessed_data_manager.get_test_dataset())
        return DataLoader(testing_dataset,self.batch_size)

    def train_model(self, padding_index, preprocessed_data_manager : ManageDataset):
        criterion = torch.nn.CrossEntropyLoss(ignore_index = padding_index)    
        step = 0
        epoch, step = self.load_checkpoint()
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.6, patience=30, verbose=True)
        while epoch < self.num_epochs:
            self.model.train()
            total_loss = 0
            train_iterator_object = self.__get_train_iterator_object(preprocessed_data_manager)
            for batch_input , batch_target in train_iterator_object:
                input_model_data, target_data = self.__move_batch_to_device(batch_input, batch_target)
                output_of_model = self.__feed_model(input_model_data, target_data)
                output_for_loss, target_for_loss = self.reshape_for_loss_function(output_of_model, target_data)
                self.optimizer.zero_grad()
                loss = criterion(output_for_loss, target_for_loss)
                total_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                loss.backward()
                self.optimizer.step()
                self.plot_tensorboard(step, loss)
                step += 1
        
            self.save_checkpoint(epoch, step)
            number_of_batch = len(list(train_iterator_object))
            avg_loss = total_loss/number_of_batch
            print(f"[Epoch {epoch} / {self.num_epochs}], average train loss : {avg_loss}")
            scheduler.step(avg_loss)
            epoch += 1
            
    def load_checkpoint(self):
        epoch = step = 0
        models = os.listdir(SAVE_MODEL_PATH)
        if(models):
            checkpoint = torch.load(SAVE_MODEL_PATH.joinpath(models[0]))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']        
            step = checkpoint['step']
        return epoch, step

  
    def plot_tensorboard(self, step : int, loss : float) -> None:
        self.writer.add_scalar("Training loss", loss, global_step=step)

    def reshape_for_loss_function(self, output_of_model: torch.tensor, 
                                  target_sequence : torch.tensor) -> tuple[torch.tensor,torch.tensor]: 
        transposed_output = output_of_model.transpose(0,1)
        transposed_target = target_sequence.transpose(0,1)
        output_for_loss = transposed_output.reshape(-1, output_of_model.shape[2])
        with torch.no_grad():
            output_debug = self.__debug_evaluation(output_for_loss)
        target_for_loss = transposed_target[1:].reshape(-1)# shift the target to calculate the loss
        return output_for_loss, target_for_loss
    
    def __move_batch_to_device(self, batch_input_data : torch.tensor , batch_target_data : torch.tensor) -> tuple[torch.tensor, torch.tensor]:
        model_input_data = batch_input_data.to(DEVICE)
        target_data = batch_target_data.to(DEVICE)
        return model_input_data, target_data
        
    def __feed_model(self, model_input_data :torch.tensor , target_data : torch.tensor) -> torch.tensor:
        target_for_model = target_data[:,:-1] #shift the target sequence for the train
        return self.model(model_input_data, target_for_model)  
    
    def save_checkpoint(self, epoch : int, step : int):
        save_path =  SAVE_MODEL_PATH.joinpath(f"seq2seq_model{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': step
            },save_path)
        self.__remove_previous_model(save_path)
    
    def __remove_previous_model(self, save_path : pathlib.Path):
        models_path = os.listdir(save_path.parent)
        models_path.remove(save_path.name)
        for model in models_path:
            try:
                os.remove(SAVE_MODEL_PATH.joinpath(model))  
            except:
                pass
    
    def evaluate_model(self, padding_index : int, preprocessed_data_manager : ManageDataset) -> float :
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss(ignore_index = padding_index)    
        total_loss = 0
        test_iterator_object = self.__get_test_iterator_object(preprocessed_data_manager)
        with torch.no_grad():
            for batch_input , batch_target in test_iterator_object:
                input_model_data, target_data = self.__move_batch_to_device(batch_input, batch_target)
                output_of_model = self.__feed_model(input_model_data, target_data)
                output_for_loss, target_for_loss = self.reshape_for_loss_function(output_of_model, target_data)
                loss = criterion(output_for_loss, target_for_loss)
                total_loss += loss.item()
            
        return total_loss / len(list(test_iterator_object))        

    def __debug_evaluation(self,output_tensor: torch.tensor):
        return torch.tensor(list(map(lambda feature_value : feature_value.argmax(), output_tensor)))
       
    
def train_object(dropout_value = 0.1, embedding_size = 64):
    dataset_manager = ManageDataset(2)
    translator_object = dataset_manager.get_translatorObject()
    pytorch_transformers_argument = PytorchTransformerArguments()
    num_head = pytorch_transformers_argument.get_arguments()["nhead"]
    inner_input_representation_model = OneHotEncoding(translator_object, num_head)
    training_hyperparameter = TrainingTransformerHyperparameters()
    model = Transformer(inner_input_representation_model,pytorch_transformers_argument)
    training_object = TrainingAndTestTransformer(**training_hyperparameter.get_arguments(model))
    padded_index = translator_object.get_padding_index()
    training_object.train_model(padded_index, dataset_manager)
    print(f"loss on test : {training_object.evaluate_model(padded_index, dataset_manager)}")
    
    