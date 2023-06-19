import BERTpreprocessing
from modelArgumentManagement import ArgumentFromJson
import numpy as np
import evaluate
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

BERT_ARGUMENT_PATH = "bert_fine_tuning_argument.json"
MODEL_DIRECTORY = "models/finetuned_bert"

class EvaluationManager():
    def __init__(self) -> None:
        self.label_list = [0,1]
        self.evaluate_object = evaluate.load('seqeval')
    
    def compute_metrics(self, model_prediction_and_label) -> dict:
        model_predictions, labels = model_prediction_and_label
        value_predictions = np.argmax(model_predictions, axis=2)

        #true_predictions = [
        #   [p for (p, l) in zip(prediction, label) if l != -100]
        #  for prediction, label in zip(value_predictions, labels)
        #]
        #true_labels = [
        #   [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        #  for prediction, label in zip(model_predictions, labels)
        #]
        results = self.evaluate_object.compute(predictions= value_predictions, references= labels)
        return results
    
    def get_metrics(self):
        return self.compute_metrics
    
class BertArgumentFromJson(ArgumentFromJson):
    def __init__(self, parameters_file_path = BERT_ARGUMENT_PATH):
        super().__init__(parameters_file_path)
    
    def get_argument(self,model_directory) -> dict:
        self.parameters_dict['output_dir'] = model_directory
        return super().get_argument()
    
class FineTuningBERTTrainer():
    def __init__(self, tokenization_object : BERTpreprocessing.DatasetTokenizer, 
                 train_test_dataset : BERTpreprocessing.DatasetPreprocessor,
                 evaluation_function : callable) -> None:
        self.train_dataset = tokenization_object.tokenize_dataset(train_test_dataset.get_train_dataset())
        self.test_dataset = tokenization_object.tokenize_dataset(train_test_dataset.get_test_dataset())
        self.tokenizer = tokenization_object.tokenizer
        self.data_collator = tokenization_object.get_data_collator()
        train_argument = BertArgumentFromJson().get_argument(MODEL_DIRECTORY)
        self.model = AutoModelForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
        self.training_arguments = TrainingArguments(**train_argument)
        self.evaluation_function = evaluation_function
    
    def train_model(self):
        trainer = Trainer(
            model = self.model,
            args = self.training_arguments,
            train_dataset = self.train_dataset,
            eval_dataset = self.test_dataset,
            tokenizer =  self.tokenizer,
            data_collator= self.data_collator ,
            compute_metrics = self.evaluation_function,
        )
        trainer.train()
        
    
    