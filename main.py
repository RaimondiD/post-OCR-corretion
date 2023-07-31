import BERTFineTuning
import trainTestSplitter
import BERTpreprocessing
import correctionSentences
import textSegmentator
import torch
import random
RANDOM_SEED = 12062022
DEFAULT_BATCH_SIZE = 32
DIMENSION_OF_SAMPLE = 3
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


    
def train_fine_tuned_BERT():
    dataset_object = trainTestSplitter.TrainTestSplitter(-1)
    train_test_dataset = BERTpreprocessing.DatasetPreprocessor(dataset_object)
    tokenization_object = BERTpreprocessing.DatasetTokenizer()
    evaluation_function = BERTFineTuning.EvaluationManager().get_metrics()
    trainingObject = BERTFineTuning.FineTuningBERTTrainer(tokenization_object, train_test_dataset, evaluation_function)
    trainingObject.train_model()

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


def correct_sentences():
    dataset_object = trainTestSplitter.TrainTestSplitter()
    segmentator = textSegmentator.TextSplitter(dataset_object.get_seq2seq_train_dataset())
    segmentator_output = segmentator.get_splitted_sentences()
    corrector = SentenceCorrector()
    corrected_sentences = corrector.get_corrected_phrases(segmentator_output)
    get_performance_report(dataset_object, segmentator, corrector)
    
    
def test_segmentator_and_corrector():
    dataset_object = trainTestSplitter.TrainTestSplitter()
    dummy_segmentator = textSegmentator.DummyTextSplitter(dataset_object)
    dummy_output = dummy_segmentator.get_splitted_sentences()
    corrector = correctionSentences.BERTSentenceCorrector(dataset_object.get_dictionary_data(), dummy_output)
    print(corrector.levensthein_distance_improvement(dataset_object.get_test_string_datasets()[1]))

test_segmentator_and_corrector()