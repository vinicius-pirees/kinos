import unittest
from kinos.models.gaussian import Gaussian
from sklearn.datasets import load_sample_images



class TestGaussian(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        dataset = load_sample_images() 
        sequence_1 = [dataset.images[0] for x in range(20)]
        sequence_2 = [dataset.images[1] for x in range(20)]
        self.sequences = [sequence_1,sequence_2]


    def test_gaussian_with_pca(self):
        gaussian = Gaussian(model_name='gaussian_1', pca=True, pca_n_components=6)
        gaussian.fit(self.sequences)
        gaussian.save_model()


    @classmethod
    def tearDownClass(cls):
        pass


if __name__ == '__main__':
    unittest.main()