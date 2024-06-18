import json

class Json_read:
    def __init__(self, parameters, results):
        self.parameters = parameters
        self.results = results
      
    # update Json from object 
    def set_json(self, parameters,results):


        self.parameters = parameters.copy()
        self.results = results
        with open('best_param.json', 'w') as file:
                json.dump([parameters,results],  file, indent=4)
              
                



    # update object from Json 
    def update_from_json(self):   
        with open('best_param.json', 'r') as file:
            json_best_param = json.load(file)
            self.parameters = json_best_param[0]
            self.results = json_best_param[1]

                  



