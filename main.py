import seq2seqTraining
import seq2seqLearning 
import seq2seqEvaluation
import BERTFineTuning
import trainTestSplitter
import BERTpreprocessing
import correctionSentences
import torch
import random
RANDOM_SEED = 12062022
DEFAULT_BATCH_SIZE = 32
DIMENSION_OF_SAMPLE = 3
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def train_and_evaluateSeq2seQ(batch_size, dimension_of_sample = -1):
    dataset_manager = seq2seqLearning.ManageDataset(dimension_of_sample)
    padding_index = dataset_manager.get_translatorObject().get_padding_index()
    criterion = torch.nn.CrossEntropyLoss(ignore_index = padding_index)    
    model = train_seq2seq(dataset_manager, criterion)
    evalute_seq2seq(dataset_manager, criterion, model, batch_size)

def train_seq2seq(dataset_manager : seq2seqLearning.ManageDataset, criterion, dropout_value = 0.001, embedding_size = 64):
    translator_object = dataset_manager.get_translatorObject()
    pytorch_transformers_argument = seq2seqLearning.PytorchTransformerArguments()
    inner_input_representation_model = seq2seqLearning.EmbeddingRepresentation(translator_object, embedding_size, dropout_value) #substitute with the desired encoding, (one hot, Embedding, concatEmbedding)
    training_hyperparameter = seq2seqTraining.TrainingTransformerHyperparameters()
    model = seq2seqLearning.Transformer(inner_input_representation_model,pytorch_transformers_argument)
    training_object = seq2seqTraining.TrainTransformer(**training_hyperparameter.get_arguments(model, criterion))
    return training_object.train_model(dataset_manager.get_train_dataset())
    

def evalute_seq2seq(dataset_manager : seq2seqLearning.ManageDataset, criterion, model : seq2seqLearning.Transformer,
                  batch_size : int):
    translator_object = dataset_manager.get_translatorObject()
    test_dataset = dataset_manager.get_test_dataset()
    test_object = seq2seqEvaluation.Sequence2SequenceEvaluator(model,batch_size, criterion, test_dataset, translator_object)
    print(test_object.evaluate_model())
    print(test_object.get_levenshtein_similarity())
    
    
def train_fine_tuned_BERT():
    dataset_object = trainTestSplitter.TrainTestSplitter(-1)
    train_test_dataset = BERTpreprocessing.DatasetPreprocessor(dataset_object)
    tokenization_object = BERTpreprocessing.DatasetTokenizer()
    evaluation_function = BERTFineTuning.EvaluationManager().get_metrics()
    trainingObject = BERTFineTuning.FineTuningBERTTrainer(tokenization_object, train_test_dataset, evaluation_function)
    trainingObject.train_model()
#train_and_evaluateSeq2seQ(DEFAULT_BATCH_SIZE,DIMENSION_OF_SAMPLE)   #istruction to run and evaluate seq2seqmodel

def get_dictionary(train_test_object : trainTestSplitter.TrainTestSplitter):
    dictionary_input = train_test_object.get_dictionary_data()
    dictionary = correctionSentences.Dictionary(dictionary_input)
    return dictionary

def test_levensthein_token():
    dataset = trainTestSplitter.TrainTestSplitter(32)
    dictionary = get_dictionary(dataset)
    example = correctionSentences.WordPattern("hello")
    print(example.get_1_levensthein_distance_words(dictionary))
    example = correctionSentences.WordPattern("drought")
    print(example.get_1_levensthein_distance_words(dictionary))


def correct_sentences(self):
    dataset_object = trainTestSplitter.TrainTestSplitter()
    dataset_manager = dataset_manager(dataset_object.get_seq2seq_dataset())
    trained_model = train_seq2seq(dataset_manager)