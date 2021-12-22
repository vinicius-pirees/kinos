import unittest
from phaino.models.mock_model import MockModel
from phaino.deploy.model_training.model_selection import assign_models_priority
from phaino.deploy.model_training.training_manager import TrainingManager



class TestTrainingManager(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        self.user_constraints = {
            "is_real_time": False,
            "minimum_efectiveness": None
        }
        
        self.models = [
            {
                "name": "model_1",
                "training_rate": 200,
                "efectiveness": 30,
                "inference_rate": 10,
                "model":  MockModel()
            },
            {
                "name": "model_2",
                "training_rate": 300,
                "efectiveness": 20,
                "inference_rate": 20,
                "model":  MockModel()
            },
            {
                "name": "model_3",
                "training_rate": 400,
                "efectiveness": 20,
                "inference_rate": 20,
                "model":  MockModel()
            }
        ]




        self.models_with_priority = [
            {
                "name": "model_1",
                "priority": 0,
                "training_rate": 200,
                "efectiveness": 30,
                "inference_rate": 10,
                "model":  MockModel()
            },
            {
                "name": "model_3",
                "priority": 1,
                "training_rate": 400,
                "efectiveness": 20,
                "inference_rate": 20,
                "model":  MockModel()
            },
            {
                "name": "model_2",
                "priority": 2,
                "training_rate": 300,
                "efectiveness": 20,
                "inference_rate": 20,
                "model":  MockModel()
            }
            
        ]
        print("setUpClass")

    def test_priority(self):
        models_with_priority = assign_models_priority(self.user_constraints, self.models)
        self.assertEqual(len(models_with_priority), 3)

    def test_adapt(self):
        training_manager = TrainingManager(self.models_with_priority)
        training_manager.adapt()

    @classmethod
    def tearDownClass(self):
        pass


if __name__ == '__main__':
    unittest.main()