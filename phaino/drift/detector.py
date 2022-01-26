


class DriftDetector():

    def __init__(self, drift_algorithm, dimensionality_reduction, learning_setting='unsupervised', base_data=None):
        self.drift_algorithm = drift_algorithm
        self.dimensionality_reduction = dimensionality_reduction
        self.learning_setting = learning_setting

        if learning_setting == 'unsupervised':
            self.update_base_data(base_data)

        self.base_data_name=None




    def update_base_data(self, base_data, base_data_name=None):
        self.dimensionality_reduction.fit(base_data)

        if base_data_name is not None:
            self.base_data_name = base_data_name

        # Todo: decide if the metadata needs to be saved
        # #save metadata
        # metadata = {
        #     "base_data_name": self.base_data_name
           
        # }

        # with open(os.path.join(path, "metadata.json"), "w") as outfile:
        #     json.dump(metadata, outfile)

    def drift_check(self, example):
        if self.learning_setting == 'unsupervised':
            example = self.dimensionality_reduction.predict(example)

        in_drift, in_warning = self.drift_algorithm.update(example)
        if in_drift:
            return True 



