FROM python:3.12-slim

RUN pip install \
    # development dependencies
    "alicebot[all]==0.11.0" aiohttp==3.12.15 pydantic==2.11.7

COPY fix/bot.py /usr/local/lib/python3.12/site-packages/alicebot/bot.py

RUN apt-get update && apt-get install -y gosu

RUN useradd --no-log-init -d /app nyabot

COPY . /app/nyabot

WORKDIR /app/nyabot

VOLUME /app/nyabot/data

ENTRYPOINT ["bash", "entrypoint.sh"]
