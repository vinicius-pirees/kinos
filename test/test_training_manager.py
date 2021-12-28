import unittest
from unittest.case import skip
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
                "model":  MockModel(15)
            },
            {
                "name": "model_2",
                "priority": 1,
                "training_rate": 400,
                "efectiveness": 20,
                "inference_rate": 20,
                "model":  MockModel(15)
            },
            {
                "name": "model_3",
                "priority": 2,
                "training_rate": 300,
                "efectiveness": 20,
                "inference_rate": 20,
                "model":  MockModel(15)
            }
            
        ]


        self.models_with_priority_different_times = [
            {
                "name": "model_1",
                "priority": 0,
                "training_rate": 200,
                "efectiveness": 30,
                "inference_rate": 10,
                "model":  MockModel(10)
            },
            {
                "name": "model_2",
                "priority": 1,
                "training_rate": 400,
                "efectiveness": 20,
                "inference_rate": 20,
                "model":  MockModel(6)
            },
            {
                "name": "model_3",
                "priority": 2,
                "training_rate": 300,
                "efectiveness": 20,
                "inference_rate": 20,
                "model":  MockModel(60)
            }
            
        ]

        print("setUpClass")

    def test_priority(self):
        models_with_priority = assign_models_priority(self.user_constraints, self.models)
        self.assertEqual(len(models_with_priority), 3)

    @skip
    def test_adapt(self):
        training_manager = TrainingManager(self.models_with_priority)
        with self.assertRaises(SystemExit) as cm:
            training_manager.adapt()

        self.assertEqual(cm.exception.code, 0)

    #@skip
    def test_adapt_different_times(self):
        training_manager = TrainingManager(self.models_with_priority_different_times)

        with self.assertRaises(SystemExit) as cm:
            training_manager.adapt()

        self.assertEqual(cm.exception.code, 0)


        
      

    @classmethod
    def tearDownClass(self):
        pass


if __name__ == '__main__':
    unittest.main()