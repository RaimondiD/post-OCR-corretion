from seq2seqPreprocessing import transform_data_to_token,IndexTranslator
from abc import abstractmethod
from sklearn.model_selection import train_test_split
import torch
import json
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

ENCODER_PARAMETER_DEFAULT_PATH = "pytorch_encoder_parameters.json"
TRAINING_PARAMETER_DEFAULT_PATH = "training_object_parameters.json"
DEVICE = 'cpu'

class Seq2SeqDataset(Dataset):
    def __init__(self, input_senquence:list[list[str]], target_sequence: list[list[str]]):
        assert(len(input_senquence) == len(target_sequence))
        self.input_sequence = torch.tensor(input_senquence)
        self.target_sequence = torch.tensor(target_sequence)

    def __len__(self):
        return self.input_sequence.size()[0]
    
    def __getitem__(self, index:int) -> torch.tensor:
        return self.input_sequence[index], self.target_sequence[index]

class ManageDataset():
    def __init__(self, last_index=-1):
        input_sequence_data, output_sequence_data, self.translatorObject = transform_data_to_token(last_index)
        self.train_input, self.test_input, self.train_output, self.test_output = \
            train_test_split(input_sequence_data,output_sequence_data)

    def get_translatorObject(self):
        return self.translatorObject
    
    def get_input(self):
        return self.train_input + self.test_input
    
    def get_output(self):
        return self.train_output + self.test_output
    
    def get_train_dataset(self):
        return self.train_input, self.train_output
    
    def get_test_dataset(self):
        return self.train_input, self.train_output
    
    def get_tensor_representation(self):
        return torch.tensor(self.get_input()), torch.tensor(self.get_output())


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

class ArgumentFromJson : 
    def __init__(self, parameters_file_path : str):
        with open(parameters_file_path,"r") as arguments_file:
            self.parameters_dict = json.load(arguments_file) 

class PytorchTransformerArguments(ArgumentFromJson):
    def __init__(self, encoder_parameters_file_path = ENCODER_PARAMETER_DEFAULT_PATH):
        super(PytorchTransformerArguments,self).__init__(encoder_parameters_file_path)

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
        self.train_iterator_object = self._get_train_iterator_object()
        self.model = model.to(DEVICE)
        self.writer = SummaryWriter("runs/loss_plot")


    def train_model(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyperparameters['learning_rate'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, verbose=True)
        criterion = torch.nn.CrossEntropyLoss()    #try to put "ignore index"
        step = 0
        for epoch in range(self.num_epochs):
            print(f"[Epoch {epoch} / {self.num_epochs}]")
            self.model.eval()
            self.model.train()
            losses = []

            for batch_input , batch_target in self.train_iterator_object:
                # Get input and targets and get to cuda
                inp_data = batch_input.to(DEVICE)
                target = batch_target.to(DEVICE)

                # Forward prop
                output = self.model(inp_data, target)

                # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
                # doesn't take input in that form. For example if we have MNIST we want to have
                # output to be: (N, 10) and targets just (N). Here we can view it in a similar
                # way that we have output_words * batch_size that we want to send in into
                # our cost function, so we need to do some reshapin.
                # Let's also remove the start token while we're at it
                output = output.reshape(-1, output.shape[2])
                target = target.reshape(-1)

                optimizer.zero_grad()

                loss = criterion(output, target)
                losses.append(loss.item())

                # Back prop
                loss.backward()
                # Clip to avoid exploding gradient issues, makes sure grads are
                # within a healthy range
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)

                # Gradient descent step
                optimizer.step()

                self.plot_tensorboard(step, loss)
                step += 1

        mean_loss = sum(losses) / len(losses)
        scheduler.step(mean_loss)

    def _get_train_iterator_object(self):
        preprocessed_data_manager = ManageDataset()
        batch_size = self.hyperparameters['batch_size']
        training_dataset = Seq2SeqDataset(*preprocessed_data_manager.get_train_dataset())
        return DataLoader(training_dataset,batch_size)

    def plot_tensorboard(self, step, loss):
        self.writer.add_scalar("Training loss", loss, global_step=step)


    def evaluate_model(self):
        pass
def train_object(dropout_value = 0.1, embedding_size = 32):
    dataset_manager = ManageDataset()
    translator_object = dataset_manager.get_translatorObject()
    inner_input_representation_model = EmbeddingRepresentation(translator_object, embedding_size, dropout_value)
    pytorch_transformers_argument = PytorchTransformerArguments()
    training_hyperparameter = TrainingTransformerHyperparameters()
    model = Transformer(inner_input_representation_model,pytorch_transformers_argument)
    training_object = TrainingAndTestTransformer(model, training_hyperparameter)
    training_object.train_model()
    
    
