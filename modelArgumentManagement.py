import json

class ArgumentFromJson : 
    def __init__(self, parameters_file_path : str):
        with open(parameters_file_path,"r") as arguments_file:
            self.parameters_dict = json.load(arguments_file) 

    def get_argument(self) -> dict :
        return self.parameters_dict
    
    