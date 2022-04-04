# Documentation

## Configuration
The framework configuration is set in the file `config/phaino.conf`.

Example:
```
[general]
profile = dev
project_name = evaluation

[dev]
directory = /tmp/test_directory
kafka_broker_list = localhost:9092

[prod]
directory = /home/user/models
kafka_broker_list = kafka-host:29092,kafka-host:29093
``` 

It is possible to configure multiple profiles. Profiles are closed environments which its settings affect only that profile.

* General section: the [general] section contains the configuration valid for all profiles:

    * profile: specifies which profile to use
    * project_name: a project name. This name defines the directory structure where the models and metadata will be saved. It also defines naming conventions of Kafka consumers. It is possible to switch back and forth between projects.

* Profiles sections: there can be more than one [profile] section. The name of the profile is used as the section name.

    * directory: defines the root directory where models and metadata for all projects will be stored.
    * kafka_broker_list: list of Kafka brokers.



## Setting up Kafka

Kafka is a distributed data platform for messages, or real-time time, thus suitable for video processing.

The use of a kafka cluster is encouraged. For development purposes, however, a docker-compose installation is provided.

Steps:
* Install Docker (https://docs.docker.com/get-docker/)

* Define where Kafka data will be stored using the environment variable KAFKA_DATA

    Ex:
    ```bash
    $ export KAFKA_DATA=~/kafka_data/
    ```

* Go to the directory ./docker (where the file docker-compose.yml is located)
    ```bash
    $ cd docker/
    ```

* Bring the cluster up
    ```bash
    $ docker-compose up -d
    ```

Kafka will be availabe at `localhost:9092`


## Data sources: training and inference
Multiple sources of data can be used. The methods to consume data from images, cameras, and videos can be used as below.


```python
import cv2
from phaino.config.config import PhainoConfiguration
from phaino.streams.producers import VideoProducer, CameraProducer, ImageProducer

config = PhainoConfiguration().get_config()
profile = config['general']['profile']
KAFKA_BROKER_LIST = config[profile]['kafka_broker_list']
TOPIC = 'inference'


# Video source
video_producer = VideoProducer(KAFKA_BROKER_LIST, TOPIC, '/home/myuser/myvideo.mp4')
video_producer.send_video(extra_fields={"sequence_name": 'myvideo'})


# Camera source
camera_number = 0
camera_producer = CameraProducer(KAFKA_BROKER_LIST, TOPIC, camera_number)
camera_producer.send_video(extra_fields={"sequence_name": 'cam_0'})


# Image source
image_producer = ImageProducer(KAFKA_BROKER_LIST, TOPIC)
image = cv2.imread('/home/myuser/myimage.png')
image_producer.send_frame(image, extra_fields={"sequence_name": 'image_1'})

```

## Models

There are predefined models at the directory phaino/models/ and other ones can be created. The models should have the following methods:
    
* fit(training_set) -> trains the model with the respective training_set passes as the argument
* predict(example) -> returns the prediction for the example argument
* save_model() -> saves the model and metadata to disk
* loads_model() -> loads the previously saved model


### Model configuration to phaino
In order to use a model with phaino it has to be configured as follows:
* model_name: name given to the model so it can be saved, loaded and audited later
* training_rate: estimate of how many examples can be trained per second
* efectiveness: estimate of effectivess. Can be any metric, but has to be consistent over all models
* inference_rate: estimate of how many examples per second it can handle at the inference phase
* model: the actual model class


Ex:
```python
{
    "name": "gaussian_1",
    "training_rate": 200,
    "efectiveness": 30,
    "inference_rate": 10,
    "model":  Gaussian(model_name='gaussian_1', pca=True, pca_n_components=100)
}
```

## User constraints
Constraints can be defined to restrict the use of models based on effectiveness and computing time goals.

Ex:
```python
user_constraints = {
    "is_real_time": False,
    "minimum_efectiveness": None
}
```

## Concept drift detection
Concept drift detection algorithms should, for every new example/value, return whether or not a drift has been detected.

Ex:

```python
in_drift, in_warning = drift_algorithm.update(example)
```

Although any drift detection algorithm can be used, the [river](https://riverml.xyz) library contains a set of drift detector that are ready to use. 

Ex:
```python
#https://riverml.xyz/latest/api/drift/PageHinkley/
from river.drift import PageHinkley
drift_algorithm = PageHinkley()


in_drift, in_warning = drift_algorithm.update(0.1)
if in_drift:
    print("Change detected!")
``` 

## Dimensionality reduction
The concept drift is done by comparing the inference data with previous data used for training. The comparison is done using a dimensionality reduction algorithm. The following methods should be implemented:
* fit(base_data) -> perform dimensionality reduction on the base_data
* predict(examples) -> return a score indicating how similiar each example is in relation to the base_data

In addition to implement your own dimensionality reduction techniques, you can also use the ones available at phaino/drift/dimensionality_reduction/. An example is PCA.

Ex:
```python
from phaino.drift.dimensionality_reduction.pca import PCA

dimensionality_reduction = PCA()
dimensionality_reduction.fit(base_data)
dimensionality_reduction.predict([0.1,0.2])
```

## Training and continuous adaptation - phaino handler

The `MainHandler` is responsible for conducting the paralell training, adaption and inference for all the models. These are the possible configurations:

* **models**

    List of models

* **user_constraints**

    Constraints that can be imposed by the user. Default={}.


* **drift_algorithm**

    Drift detection algorithm.

* **dimensionality_reduction**

    Dimensionality reduction algorithm.

* **training_data_topic**

    Name of the Kafka topic where the training data is located

* **is_initial_training_from_topic**

    Obtain the initial training data from the kafka topic. Default=True

* **initial_training_data**

    The initial traning data to use when is_initial_training_from_topic=False. 

* **inference_data_topic**

    Name of the Kafka topic where the inference data is located

* **prediction_result_topic**

    Name of the Kafka topic where to store the predictions

* **drift_alert_topic**

    Name of the Kafka topic where to store the concept drift detections

* **frame_dimension**

    Final output frame dimension. Default=(256,256)

* **read_retries**

    Number of retries when reading the traning data. Default=2.

* **initially_load_models**

    Wheter or not to use previously trained models. Default=False.

* **detect_drift**

    Whether or not to detect concept drifts. Default=True.

* **adapt_after_drift**

    Whether or not to adapt the models after a concept drift is detected. Default=True.

* **number_training_frames_after_drift**

    (Unsupervised tasks) Number of frames to use as training after the drift is detected. Default=200.

* **provide_training_data_after_drift**

    (Supervised tasks) Option to provide the training data manually after a concept drift is detected. Default=False.



## End-end-end example
```python
from phaino.deploy.main_handler import MainHandler
from river.drift import PageHinkley
from phaino.drift.dimensionality_reduction.pca import PCA

from phaino.models.gaussian import Gaussian
from phaino.models.lstm_autoencoder import LSTMAutoEncoder
from phaino.models.oneclass_svm import OneClassSVM

inference_data_topic = 'inference'
prediction_result_topic = 'prediction'
training_data_topic = 'training'
drift_alert_topic = 'drift'


user_constraints = {
    "is_real_time": False,
    "minimum_efectiveness": None
}

models = [
    {
        "name": "gaussian_1",
        "training_rate": 200,
        "efectiveness": 30,
        "inference_rate": 10,
        "model":  Gaussian(model_name='gaussian_1', pca=True, pca_n_components=100)
    },
    {
        "name": "lstm_1",
        "training_rate": 30,
        "efectiveness": 60,
        "inference_rate": 3,
        "model":  LSTMAutoEncoder(model_name='lstm_1', epochs=3)
    },
    {
        "name": "oneclass_svm_1",
        "training_rate": 220,
        "efectiveness": 20,
        "inference_rate": 10,
        "model":  OneClassSVM(model_name='oneclass_svm_1', pca=False)
    },
]

drift_algorithm = PageHinkley()
dimensionality_reduction = PCA()
number_training_frames_after_drift = 500



handler = MainHandler(
            models=models,
            user_constraints=user_constraints,
            number_training_frames_after_drift=number_training_frames_after_drift,
            drift_algorithm=drift_algorithm,
            dimensionality_reduction=dimensionality_reduction,
            training_data_topic=training_data_topic,
            prediction_result_topic=prediction_result_topic,
            inference_data_topic=inference_data_topic,
            drift_alert_topic=drift_alert_topic
            )
```

