from  seq2seqLearning import TrainTestTransformer, Transformer
from modelArgumentManagement import ArgumentFromJson
import os
import pathlib
from torch.utils.tensorboard import SummaryWriter
import torch

TRAINING_PARAMETER_DEFAULT_PATH = pathlib.Path("training_object_parameters.json")
SAVE_MODEL_PATH = pathlib.Path("models/test_checkpoint")
NAME_OF_SAVED_MODEL = pathlib.Path("seq2seq_test_model")
TENSORBOARD_RESULT = pathlib.Path("models/tensorboard/seq2seq_tran_loss")




class TrainingTransformerHyperparameters(ArgumentFromJson):
    def __init__(self, training_transformer_hyperparameters_path = TRAINING_PARAMETER_DEFAULT_PATH):
        super(TrainingTransformerHyperparameters,self).__init__(training_transformer_hyperparameters_path)
    
    def get_arguments(self, model : Transformer, criterion) -> dict :
        self.parameters_dict['model'] = model
        self.parameters_dict['criterion'] = criterion
        return self.parameters_dict
      
    
class TrainTransformer(TrainTestTransformer) :
    def __init__(self, model: Transformer, learning_rate: float, batch_size: float, num_epochs: int, criterion) -> None:
        super().__init__(model, batch_size, criterion)
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.writer = SummaryWriter(TENSORBOARD_RESULT)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = learning_rate)

    def train_model(self, train_dataset : tuple[torch.tensor, torch.tensor]):
        step = 0
        epoch, step = self.load_checkpoint()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.1, patience=5, verbose=True)
        while epoch < self.num_epochs:
            self.model.train()
            total_loss = 0
            train_iterator_object = self.get_dataset_iterator_object(train_dataset)
            for batch_input , batch_target in train_iterator_object:
                input_model_data, target_data = self.move_batch_to_device(batch_input, batch_target)
                output_of_model = self.feed_model(input_model_data, target_data)
                output_for_loss, target_for_loss = self.reshape_for_loss_function(output_of_model, target_data)
                self.optimizer.zero_grad()
                loss = self.criterion(output_for_loss, target_for_loss)
                total_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                loss.backward()
                self.optimizer.step()
                self.__plot_tensorboard(step, loss)
                step += 1
                    
            self.save_checkpoint(epoch, step)
            number_of_batch = len(list(train_iterator_object))
            avg_loss = total_loss/number_of_batch
            print(f"[Epoch {epoch} / {self.num_epochs}], average train loss : {avg_loss}")
            scheduler.step(avg_loss)
            epoch += 1
            
        return self.model
    
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
    
    def __plot_tensorboard(self, step : int, loss : float) -> None:
            self.writer.add_scalar("Training loss", loss, global_step=step)
            