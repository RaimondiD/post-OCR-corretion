import unittest
import seq2seqLearning
from seq2seqPreprocessing import transform_data_to_token, IndexTranslator
import torch

DIMENSION_OF_SAMPLE = 8
DROPOUT_RATE = 0.1
EMBEDDING_SIZE = 32

class TestEmbedding(unittest.TestCase):

    def __init__(self,*args,**kwargs) -> None:
   
        dataset_info = seq2seqLearning.ManageDataset(DIMENSION_OF_SAMPLE)
        self.input_tensor, self.output_tensor = dataset_info.get_tensor_representation()
        self.dataset_translator_object = dataset_info.get_translatorObject()
        self.internal_representation_object = seq2seqLearning.EmbeddingRepresentation(self.dataset_translator_object
                                                                                      , EMBEDDING_SIZE,
                                                                                      DROPOUT_RATE)
        unittest.TestCase.__init__(self,*args,**kwargs)        

    def test_real_data_repr_size(self):
        self.assertEqual(self.dataset_translator_object.get_vocabolary_dimension(), self.internal_representation_object.get_data_original_size())

    def test_data_repr_size(self):
        self.assertTrue(self.internal_representation_object.get_data_representation_size() == EMBEDDING_SIZE)
    
    def test_data_representation(self):
        input_embedding = self.internal_representation_object.forward(self.input_tensor[:2])
        output_embedding = self.internal_representation_object.forward(self.output_tensor[:2])
        self.assertTrue(input_embedding.size() <= output_embedding.size(),f"input_tensor : {self.input_tensor.size()}, output_tenso : {self.output_tensor.size()}")

    def test_padding_index(self):
        self.assertEqual(self.dataset_translator_object.get_padding_index(), self.internal_representation_object.padding_index())


class testTransformer(unittest.TestCase):
    def __init__(self,*args,**kwargs) -> None:
        dataset_info = seq2seqLearning.ManageDataset(DIMENSION_OF_SAMPLE)
        self.input_tensor, self.output_tensor = dataset_info.get_tensor_representation()
        input_dictionary = dataset_info.get_translatorObject()
        input_repr_object = seq2seqLearning.EmbeddingRepresentation(input_dictionary, EMBEDDING_SIZE, DROPOUT_RATE)
        transformer_argument = seq2seqLearning.PytorchTransformerArguments()
        self.myTransformer = seq2seqLearning.Transformer(input_repr_object, transformer_argument)
        unittest.TestCase.__init__(self,*args,**kwargs)        
            

    def test_input_mask(self):
        padding_mask_result = self.myTransformer.get_padding_mask(self.input_tensor)
        print(f"""sample_vector: {padding_mask_result[0]}
               shape of mask : {padding_mask_result.size()}
               shape of input : {self.input_tensor.size()}""")
        self.assertEqual(self.myTransformer.get_padding_mask(self.input_tensor).size() , self.input_tensor.size())

    def test_forward(self):
        print(self.input_tensor.size(),self.output_tensor.size())
        self.myTransformer.forward(self.input_tensor, self.output_tensor)

class testDatasetManagement(unittest.TestCase):
    def __init__(self,*args,**kwargs):
        self.dataset_management = seq2seqLearning.ManageDataset(DIMENSION_OF_SAMPLE)
        unittest.TestCase.__init__(self,*args,**kwargs)        
    
    def test_preprocessed_dataset(self):
        input_dataset, output_dataset, _ = transform_data_to_token(DIMENSION_OF_SAMPLE)
        input_from_class = self.dataset_management.get_input()
        output_from_class = self.dataset_management.get_output()
        self.assertEqual(len(input_from_class), len(input_dataset))
        self.assertEqual(len(output_from_class), len(output_dataset))


class testTrainingObject(unittest.TestCase):
    def test_training(self):
        seq2seqLearning.train_object()
    


if __name__ == '__main__':
    unittest.main()