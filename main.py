import seq2esqTraining
import seq2seqLearning 
import seq2seqEvaluation
import torch
import random
RANDOM_SEED = 12062022
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

def train_object(dataset_manager : seq2seqLearning.ManageDataset, criterion, dropout_value = 0.001, embedding_size = 64):
    translator_object = dataset_manager.get_translatorObject()
    pytorch_transformers_argument = seq2seqLearning.PytorchTransformerArguments()
    inner_input_representation_model = seq2seqLearning.EmbeddingRepresentation(translator_object, embedding_size, dropout_value) #substitute with the desired encoding, (one hot, Embedding, concatEmbedding)
    training_hyperparameter = seq2esqTraining.TrainingTransformerHyperparameters()
    model = seq2seqLearning.Transformer(inner_input_representation_model,pytorch_transformers_argument)
    padded_index = translator_object.get_padding_index()
    training_object = seq2esqTraining.TrainTransformer(**training_hyperparameter.get_arguments(model, criterion))
    return training_object.train_model(dataset_manager.get_train_dataset())
    

def evalute_model(dataset_manager : seq2seqLearning.ManageDataset, criterion, model : seq2seqLearning.Transformer,
                  batch_size : int):
    translator_object = dataset_manager.get_translatorObject()
    test_dataset = dataset_manager.get_train_dataset()
    test_object = seq2seqEvaluation.Sequence2SequenceEvaluator(model,batch_size, criterion, test_dataset, translator_object)
    print(test_object.evaluate_model())
    print(test_object.get_levenshtein_similarity())