import unittest
from unittest.case import skip
from kinos.models.gaussian import Gaussian
from kinos.models.mock_model import MockModel
from kinos.deploy.model_training.model_selection import assign_models_priority
from kinos.deploy.model_training.training_manager import TrainingManager



class TestTrainingManager(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        self.user_constraints = {
            "is_real_time": False,
            "minimum_efectiveness": None
        }
        
        self.models = [
            {
                "name": "gaussian_1",
                "training_rate": 200,
                "efectiveness": 30,
                "inference_rate": 10,
                "model":  Gaussian(model_name='gaussian_1', pca=True, pca_n_components=.95)
            },
            {
                "name": "gaussian_2",
                "training_rate": 250,
                "efectiveness": 25,
                "inference_rate": 10,
                "model":  Gaussian(model_name='gaussian_2', pca=True, pca_n_components=.90)
            }
        ]

  
        print("setUpClass")

    @skip
    def test_priority(self):
        #TODO actually verify the priorities, test with constraints
        training_manager = TrainingManager(self.models, self.user_constraints)
        self.assertEqual(len(training_manager.models), 2)

    
    def test_adapt(self):
        training_manager = TrainingManager(self.models)
        # with self.assertRaises(SystemExit) as cm:
        #     training_manager.adapt()

        # self.assertEqual(cm.exception.code, 0)
        self.assertEqual(training_manager.adapt(load_models=True), 0) 

    
      

    @classmethod
    def tearDownClass(self):
        pass


if __name__ == '__main__':
    unittest.main()