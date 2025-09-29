FROM python:3.12-slim

RUN pip install \
    # development dependencies
    "alicebot[all]==0.11.0" aiohttp==3.12.15 pydantic==2.11.7 \
    # others
    black isort

COPY fix/bot.py /usr/local/lib/python3.12/site-packages/alicebot/bot.py
