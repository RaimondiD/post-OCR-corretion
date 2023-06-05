import unittest
import seq2seqLearning
from seq2seqPreprocessing import transform_data_to_token, IndexTranslator
import torch

def instantiate_embedding(tokenized_data_object :IndexTranslator, test_word_embedding_size = 32, test_dropout_value = 0.1):
    input, output, dictionary = tokenized_data_object
    input_tensor = torch.tensor(input, dtype = torch.int )
    output_tensor = torch.tensor(output, dtype = torch.int)
    embedding =  seq2seqLearning.EmbeddingRepresentation(dictionary,
                                                        test_word_embedding_size,
                                                        test_dropout_value)
    return embedding, input_tensor, output_tensor

class TestEmbedding(unittest.TestCase):

    
    def __init__(self,*args,**kwargs) -> None:
        input,output,self.dictionary = transform_data_to_token(20)
        self.test_word_embedding_size = 30
        self.test_dropout_valute = 0.1
        self.input_representation_obj, self.input_tensor, self.output_tensor = instantiate_embedding(
                                        tokenized_data_object=(input, output, self.dictionary),
                                        test_word_embedding_size=self.test_word_embedding_size,
                                        test_dropout_value = self.test_dropout_valute
                                        )
        unittest.TestCase.__init__(self,*args,**kwargs)        

    def test_real_data_repr_size(self):
        self.assertEqual(self.dictionary.get_vocabolary_dimension(), self.input_representation_obj.get_data_original_size())

    def test_data_repr_size(self):
        self.assertTrue(self.input_representation_obj.get_data_representation_size() == self.test_word_embedding_size)
    
    def test_data_representation(self):
        input_embedding = self.input_representation_obj.get_data_representation(self.input_tensor[:2])
        output_embedding = self.input_representation_obj.get_data_representation(self.output_tensor[:2])
        self.assertTrue(input_embedding.size()!=output_embedding.size(),f"input_tensor : {self.input_tensor.size()}, output_tenso : {self.output_tensor.size()}")

    def test_padding_index(self):
        self.assertEqual(self.dictionary.get_padding_index(), self.input_representation_obj.padding_index())


class testTransformer(unittest.TestCase):
    def __init__(self,*args,**kwargs) -> None:
        tokenized_data_object = transform_data_to_token(20)
        input_repr_object, self.input_tensor, self.output_tensor = instantiate_embedding(tokenized_data_object)
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
        self.myTransformer(self.input_tensor, self.output_tensor)

if __name__ == '__main__':
    unittest.main()