# Recuerements

<https://cloudkul.com/blog/apache-kafka-implementation/>
Dependencies for flask
sudo apt update
sudo apt upgrade
sudo apt install default-jdk
sudo apt install zookeeperd -y
sudo useradd -d /kafka -s /bin/bash kafka
cd /opt
sudo wget <https://downloads.apache.org/kafka/2.4.1/kafka_2.13-2.4.1.tgz>
mkdir -p /kafka
sudo tar -xf kafka_2.13-2.4.1.tgz -C /kafka --strip-components=1
sudo chown -R kafka:kafka /kafka
sudo su - kafka
vim config/server.properties

`pip install -U faust[eventlet]`

## Run aplication

`faust -A app_kafka worker -l info`
`$home/serg/bin/zookeeper-server-start $home/serg/etc/kafka/zookeeper.properties`

## Useful

sudo killall -TERM -u kafka
sudo lsof -i :2181

## Developing

`python setup.py develop`
`python setup.py develop`


