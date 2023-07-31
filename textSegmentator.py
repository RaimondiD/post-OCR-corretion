import seq2seqEvaluation
import trainTestSplitter
import seq2seqLearning
import seq2seqTraining
import torch

class TextSplitter():
    def get_splitted_sentences(self) -> list[str]:
        pass
    
    def levenshtein_distance_improvement(self) -> tuple[float,float]:
        pass # return two levenshtein distances, the first is of the output of the model from the output dataset (ground truth) 
             #and the second is the distance of the input from the output.
    
class DummyTextSplitter(TextSplitter):
    def __init__(self, dataset : trainTestSplitter.TrainTestSplitter):
        self.input_data, self.output_data = dataset.get_test_string_datasets()
        
    def get_splitted_sentences(self) -> list[str]:
        return self.input_data
    
    def levenshtein_distance_improvement(self) -> tuple[float]:
        return (seq2seqEvaluation.avg_levenshtein_ratio(self.input_data, self.output_data),
                seq2seqEvaluation.avg_levenshtein_ratio(self.input_data, self.output_data))


class Seq2SeqTextSplitter(TextSplitter):
    def __init__(self, dataset : trainTestSplitter.TrainTestSplitter , batch_size = 32) -> None:
        self.input_phrases, self.output_phrases = dataset.get_test_string_datasets()
        self.datasetManager = seq2seqLearning.ManageDataset(dataset.get_seq2seq_train_dataset(), dataset.get_seq2seq_test_dataset())
        self.translator_object = self.datasetManager.get_translatorObject()
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index = self.translator_object.get_padding_index())    
        model = self._train_seq2seq()
        self.evaluator = seq2seqEvaluation.Sequence2SequenceEvaluator(model, 
                                                                      batch_size,
                                                                      self.criterion,
                                                                      self.datasetManager.get_test_dataset(),
                                                                      self.translator_object
                                                                      )
        self.cached_prediction = None
    
    def get_splitted_sentences(self) -> list[str]:
        if (not self.cached_prediction):
            self.cached_prediction = self.evaluator.get_output_of_model()[0]
        predicted_segmentated_sentece = [seq2seqEvaluation.convert_list_to_string(sequence) for sequence in self.cached_prediction]
        return predicted_segmentated_sentece

        
    def _train_seq2seq(self, dropout_value = 0.001, embedding_size = 64):
        pytorch_transformers_argument = seq2seqLearning.PytorchTransformerArguments()
        inner_input_representation_model = seq2seqLearning.EmbeddingRepresentation(self.translator_object, embedding_size, dropout_value) #substitute with the desired encoding, (one hot, Embedding, concatEmbedding)
        training_hyperparameter = seq2seqTraining.TrainingTransformerHyperparameters()
        model = seq2seqLearning.Transformer(inner_input_representation_model,pytorch_transformers_argument)
        training_object = seq2seqTraining.TrainTransformer(**training_hyperparameter.get_arguments(model, self.criterion))
        return training_object.train_model(self.datasetManager.get_train_dataset())

    def levenshtein_distance_improvement(self) -> tuple[float, float]:
        return (seq2seqEvaluation.avg_levenshtein_ratio(self.input_phrases, self.output_phrases), 
                seq2seqEvaluation.avg_levenshtein_ratio(self.get_splitted_sentences(), self.output_phrases))

def evalute_seq2seq(dataset_manager : seq2seqLearning.ManageDataset, criterion, model : seq2seqLearning.Transformer,
                  batch_size : int):
    translator_object = dataset_manager.get_translatorObject()
    test_dataset = dataset_manager.get_test_dataset()
    test_object = seq2seqEvaluation.Sequence2SequenceEvaluator(model,batch_size, criterion, test_dataset, translator_object)
    print(f"loss_on_model_on_test_dataset : {test_object.evaluate_model()}, levenshtein_distance of the result : {test_object.get_levenshtein_similarity()}")
    