# Phaino

Phaino is a concept drift-aware framework for managing machine learning models applied in videos.




### Set up Kafka with docker
Define where Kafka data will be stored using the environment variable KAFKA_DATA

Ex:
```bash
$ export KAFKA_DATA=~/kafka_data/
```



Bring the cluster up
```bash
$ docker-compose up -d
```






### Errors


* Python.h File not found:
    ```bash
    sudo apt-get install python-dev
    sudo apt-get install python3-dev 
    sudo apt-get install python3.x-dev  # Where x is your python version
    ```

* fatal error: librdkafka/rdkafka.h: File or directory not found.
    Install librdkafka from confluent (https://docs.confluent.io/platform/current/installation/installing_cp/)
    ```bash

    $ wget -qO - https://packages.confluent.io/deb/7.0/archive.key | sudo apt-key add -
    $ sudo add-apt-repository "deb [arch=amd64] https://packages.confluent.io/deb/7.0 stable main"
    $ sudo add-apt-repository "deb https://packages.confluent.io/clients/deb $(lsb_release -cs) main"

    $ sudo apt install librdkafka-dev
    ```