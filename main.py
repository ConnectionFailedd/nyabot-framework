#! /usr/bin/env python

from typing import Any, Literal, Optional
import asyncio

from alicebot import Adapter, Bot
from alicebot.adapter.cqhttp import CQHTTPAdapter
from alicebot.adapter.cqhttp.message import CQHTTPMessageSegment
from alicebot.message import BuildMessageType
from structlog import get_logger

from nyaplugin.nyaplugin_base import NYAPlugin

logger = get_logger(source="main")


bot = Bot()


@bot.bot_run_hook
async def load_state(bot: Bot):
    logger.info("Bot run hook: load_state", bot=bot)
    for plugin in NYAPlugin.registered_nyaplugin_list:
        logger.info("Loading state: ", bot=bot, plugin=plugin.__name__)

        await plugin.load_state(bot)


@bot.bot_exit_hook
async def save_state(bot: Bot):
    logger.info("Bot exit hook: save_state", bot=bot)
    for plugin in NYAPlugin.registered_nyaplugin_list:
        logger.info("Saving state: ", bot=bot, plugin=plugin.__name__)

        await plugin.save_state(bot)


async def send_after_websocket_established(
    adapter: CQHTTPAdapter,
    message: BuildMessageType[CQHTTPMessageSegment],
    message_type: Literal["private", "group"],
    id_: int,
):
    while adapter.websocket is None:
        await asyncio.sleep(0.001)

    await adapter.send(message, message_type, id_)


@bot.adapter_run_hook
async def start_report(adapter: Adapter[Any, Any]):
    logger.info("Adapter run hook: start_report", adapter=adapter)
    if not isinstance(adapter, CQHTTPAdapter):
        logger.info("Adapter is not CQHTTPAdapter, skip", adapter=adapter)
        return

    bot_report_gid: Optional[int] = getattr(bot.config.bot, "bot_report_gid", None)
    if not bot_report_gid:
        logger.info(
            "Bot report failed: bot_report_gid not set",
            bot=adapter.bot,
            adapter=adapter,
        )
        return

    asyncio.create_task(
        send_after_websocket_established(
            adapter, "我，堂堂复活！", "group", bot_report_gid
        )
    )

# ! SIGTERM 信号来了之后先调用 cancel，这时 aiohttp 的 transport 会进入关闭状态，导致发送失败
# @bot.adapter_shutdown_hook
# async def exit_report(adapter: Adapter[Any, Any]):
#     logger.info("Adapter run hook: exit_report", adapter=adapter)
#     if not isinstance(adapter, CQHTTPAdapter):
#         logger.info("Adapter is not CQHTTPAdapter, skip", adapter=adapter)
#         return

#     bot_report_gid: Optional[int] = getattr(bot.config.bot, "bot_report_gid", None)
#     if not bot_report_gid:
#         logger.info(
#             "Bot report failed: bot_report_gid not set",
#             bot=adapter.bot,
#             adapter=adapter,
#         )
#         return
 
#     await adapter.send("呃啊～", "group", bot_report_gid)


def main() -> None:
    bot.run()


if __name__ == "__main__":
    main()
