#!/bin/bash
set -e

cmd="$@"

until nc -vz ${KAFKA_BOOSTRAP_SERVER_NAME} ${KAFKA_BOOSTRAP_SERVER_PORT}; do
  echo >&2 "Waiting for Kafka to be ready... - sleeping"
  sleep 2
done

echo >&2 "Kafka is up - executing command"

echo "Executing command ${cmd}"
exec $cmd
