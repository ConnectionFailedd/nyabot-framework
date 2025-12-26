#!/bin/bash

if ! docker network inspect nyabot-netbridge >/dev/null 2>&1; then
    docker network create nyabot-netbridge
fi

echo "UID=$(id -u)" > .env
echo "GID=$(id -g)" >> .env

docker compose up -d