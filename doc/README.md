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



# Setting up Kafka

Kafka is a distributed data platform for messages, or real-time time, thus suitable for video processing.

The use of a kafka cluster is encouraged. For development purposes, however, a docker-compose installation is provided.

Steps:
* download docker
* docker-compose up -d

Kafka will be availabe at `localhost:9092`


## Data sources: training and inference
Multiple sources of data can be used. For convenience, the methods to consume data from images and videos are implemented.


```python
import phaino
```

## Training and continuos adaptation

```python
import phaino
```

## Drift detection

```python
import phaino
```

## End-end-end example
```python
import phaino
```

