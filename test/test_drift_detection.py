import unittest
from phaino.drift.detector import  DriftDetector
from phaino.drift.dimensionality_reduction.pca import PCA
from phaino.drift.detector import DriftDetector
from river.drift import PageHinkley
from sklearn.datasets import load_sample_images




class TestDriftDetection(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        dataset = load_sample_images() 
        self.sequence_1 = [dataset.images[0] for x in range(20)]
        self.sequence_2 = [dataset.images[1] for x in range(20)]

        self.training_data = self.sequence_1
        self.test_data = self.sequence_1 + self.sequence_1 + self.sequence_1 + self.sequence_2


    def test_drift_pca_pagehinkley(self):
        drift_algorithm = PageHinkley(min_instances=30, delta=0.005, threshold=80, alpha=1 - 0.01)
        dimesionality_reduction = PCA()
        detector = DriftDetector(drift_algorithm, dimesionality_reduction)
        detector.update_base_data(self.training_data)

        in_drift, drift_index = detector.drift_check(self.test_data)
        print("in_drift", str(in_drift))
        print("drift_index", drift_index)

    @classmethod
    def tearDownClass(cls):
        pass


if __name__ == '__main__':
    unittest.main()