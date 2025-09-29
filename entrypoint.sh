#!/bin/bash

: ${NYABOT_GID:=0}
: ${NYABOT_UID:=0}
usermod -o -u ${NYABOT_UID} nyabot
groupmod -o -g ${NYABOT_GID} nyabot
usermod -g ${NYABOT_GID} nyabot
chown -R ${NYABOT_UID}:${NYABOT_GID} /app

# 使用 exec 才能保证最后的 PID 1 是 python main.py
exec gosu nyabot python ./main.py