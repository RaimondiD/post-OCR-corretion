from seq2seqLearning import Transformer, TrainTestTransformer
from seq2seqPreprocessing import IndexTranslator
import torch
import Levenshtein



class Sequence2SequenceEvaluator(TrainTestTransformer):
    def __init__(self, model : Transformer, 
                 batch_size : int,
                 criterion,
                 test_sequences : tuple[list[list[str]],list[list[str]]],
                 translator_object : IndexTranslator):
        super(Sequence2SequenceEvaluator,self).__init__(model, batch_size, criterion)
        self.model = model
        self.test_input_sentence, self.test_target_sentence = test_sequences
        self.translator_object = translator_object

    def get_levenshtein_similarity(self):
        natural_language_output_sequence, natural_language_target_sequence = self.get_output_of_model()
        avg_levrnshtein_distance_rate = avg_levenshtein_ratio(natural_language_output_sequence, 
                                                            natural_language_target_sequence)
        print(f"the test dataset have an average levenshtein distance rate of {avg_levrnshtein_distance_rate}")
        
        
    def get_output_of_model(self):
        self.model.eval()
        test_iterator_object = self.get_dataset_iterator_object((self.test_input_sentence, self.test_target_sentence))
        total_output = []
        total_target = []
        with torch.no_grad():
            for batch_input , batch_target in test_iterator_object:
                input_model_data, target_data = self.move_batch_to_device(batch_input, batch_target)
                output_of_model = self.feed_model(input_model_data, target_data)
                output_for_batch = list(map(lambda batch_seq : self._get_sequence_from_model_output(batch_seq),output_of_model))
                total_target += batch_target[:,1:].tolist()
                total_output += output_for_batch
        return self.list_of_list_from_encoded_to_natural_form(total_output), self.list_of_list_from_encoded_to_natural_form(total_target)
    
    def _get_sequence_from_model_output(self, model_output_batch_tensor : torch.tensor):
        output_sequence_batch_tensor = list(map(lambda tensor_sequence : tensor_sequence.argmax().item(), model_output_batch_tensor))
        return output_sequence_batch_tensor 
    
    def list_of_list_from_encoded_to_natural_form(self, encoded_list_of_sentence):
        return list(map( lambda encoded_sentence : self.translator_object.sequence_from_encode(encoded_sentence),encoded_list_of_sentence))
        
    
    def evaluate_model(self) -> float :
        self.model.eval()
        total_loss = 0
        test_iterator_object = self.get_dataset_iterator_object((self.test_input_sentence, self.test_target_sentence))
        with torch.no_grad():
            for batch_input , batch_target in test_iterator_object:
                input_model_data, target_data = self.move_batch_to_device(batch_input, batch_target)
                output_of_model = self.feed_model(input_model_data, target_data)
                output_for_loss, target_for_loss = self.reshape_for_loss_function(output_of_model, target_data)
                output_for_batch = self._get_sequence_from_model_output(output_for_loss)

                loss = self.criterion(output_for_loss, target_for_loss)
                total_loss += loss.item()
        return total_loss / len(list(test_iterator_object))        


def avg_levenshtein_ratio(model_output_encoded_sequences : list[list[str]], target_encoded_sequences : list[list[str]]) -> float:
    sum_of_levensthein_ratio = 0
    number_of_sequences = len(model_output_encoded_sequences)
    
    for model_sequence, target_encoded_sequence in zip(model_output_encoded_sequences, target_encoded_sequences):
        sum_of_levensthein_ratio += Levenshtein.ratio(model_sequence, target_encoded_sequence, processor = convert_list_to_string)
    
    return sum_of_levensthein_ratio / number_of_sequences
        
def convert_list_to_string(sequence_of_char : list[str]) -> str :
    string_of_char = "".join(sequence_of_char)
    return string_of_char


