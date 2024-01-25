import BERTFineTuning
import trainTestSplitter
import BERTpreprocessing
import correctionSentences
import textSegmentator
import torch
import random
RANDOM_SEED = 12062022
DEFAULT_BATCH_SIZE = 32
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

def train_seq2seq():
    dataset = trainTestSplitter.TrainTestSplitter()
    seq2seq_segmentator = textSegmentator.Seq2SeqTextSplitter(dataset)
    seq2seq_segmentator.evalute_seq2seq()

    
def train_fine_tuned_BERT():
    dataset_object = trainTestSplitter.TrainTestSplitter()
    train_test_dataset = BERTpreprocessing.DatasetPreprocessor(dataset_object)
    tokenization_object = BERTpreprocessing.DatasetTokenizer()
    evaluation_function = BERTFineTuning.EvaluationManager().get_metrics()
    trainingObject = BERTFineTuning.FineTuningBERTTrainer(tokenization_object, train_test_dataset, evaluation_function)
    trainingObject.train_model()


def correct_sentences():
    dataset_object = trainTestSplitter.TrainTestSplitter()
    segmentator = textSegmentator.DummyTextSplitter(dataset_object)
    segmentator_output = segmentator.get_splitted_sentences()
    corrector = correctionSentences.BERTSentenceCorrector(dataset_object.get_dictionary_data()[:50], segmentator_output[:50])
    print(corrector.levensthein_distance_improvement(dataset_object.get_test_string_datasets()[1][:50]))

correct_sentences()