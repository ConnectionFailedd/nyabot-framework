FROM python:3.12-slim

RUN pip install "nonebot2[websockets]" "nonebot-adapter-onebot" "structlog"

RUN apt-get update && apt-get install -y gosu

RUN useradd --no-log-init -d /app nyabot

COPY . /app/nyabot

WORKDIR /app/nyabot

VOLUME /app/nyabot/data

ENTRYPOINT ["bash", "entrypoint.sh"]