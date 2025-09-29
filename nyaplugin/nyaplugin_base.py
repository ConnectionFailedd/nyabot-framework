import json
import os
from abc import ABC
from typing import Any, Optional, TypeVar, cast, final

from alicebot import Bot, Event, Plugin
from alicebot.adapter.cqhttp import CQHTTPAdapter
from alicebot.config import PluginConfig
from structlog import get_logger

EventT = TypeVar("EventT", bound=Event[CQHTTPAdapter])  # 不是 CQHTTP 的适配器我不做
StateT = TypeVar("StateT", bound="NYAPluginState")
ConfigT = TypeVar("ConfigT", bound="NYAPluginConfig")


class NYAPluginState:
    """
    NYAPlugin 状态类。

    重写 from_dict() 和 to_dict() 以加载/存储状态。
    """

    async def from_dict(self, obj: Any) -> None:
        return

    async def to_dict(self) -> Any:
        return None


class NYAPluginConfig(PluginConfig):
    # 状态存储文件名。为 None 则不会加载/存储该插件状态。
    state_filename: Optional[str] = None

    # 上报群 ID，可以上报一些调试信息，不用进入后台查看。为 None 会丢弃所有上报内容。
    report_gid: Optional[int] = None


class NYAPlugin(Plugin[EventT, StateT, ConfigT], ABC):
    logger = get_logger(plugin="NYAPlugin")

    registered_nyaplugin_list: list[type["NYAPlugin[Any, Any, Any]"]] = []

    @final
    @classmethod
    async def load_state(cls, bot: Bot) -> bool:
        # 读取配置
        config = getattr(bot.config.plugin, cls.Config.__config_name__, None)
        if not config:
            cls.logger.info("Config not found")
            return False

        # 读取状态存储文件名
        state_filename = cast(NYAPluginConfig, config).state_filename
        if not state_filename:
            cls.logger.info("State filename not set")
            return False
        state_store_path = f"data/{state_filename}"

        # 初始化状态
        if not bot.plugin_state[cls.__name__]:
            bot.plugin_state[cls.__name__] = cls().__init_state__()
        if not bot.plugin_state[cls.__name__]:
            # __init_state__ 返回了 None
            cls.logger.info("State not initialized")
            return False

        # 读取状态
        if not os.path.exists(state_store_path):
            cls.logger.info("State file not found")
            return False
        with open(state_store_path, "r") as f:
            await cast(NYAPluginState, bot.plugin_state[cls.__name__]).from_dict(
                json.load(f)
            )
        cls.logger.info("State loaded")

        return True

    @final
    @classmethod
    async def save_state(cls, bot: Bot) -> bool:
        # 读取配置
        config = getattr(bot.config.plugin, cls.Config.__config_name__, None)
        if not config:
            cls.logger.info("Config not found")
            return False

        # 读取状态存储文件名
        state_filename = cast(NYAPluginConfig, config).state_filename
        if not state_filename:
            cls.logger.info("State filename not set")
            return False

        state_store_path = f"data/{state_filename}"

        # 初始化状态
        if not bot.plugin_state[cls.__name__]:
            bot.plugin_state[cls.__name__] = cls().__init_state__()
        if not bot.plugin_state[cls.__name__]:
            # __init_state__ 返回了 None
            cls.logger.info("State not initialized")
            return False

        # 写入状态
        if os.path.exists(state_store_path):
            cls.logger.info("State file backed up")
            os.rename(state_store_path, state_store_path + ".bak")
        with open(state_store_path, "w") as f:
            json.dump(
                await cast(NYAPluginState, bot.plugin_state[cls.__name__]).to_dict(), f
            )
        cls.logger.info("State saved")

        return True

    @final
    async def report(self, message: str) -> None:
        if not self.config.report_gid:
            self.logger.info("Report group not specified, discard report info")
            return

        await self.event.adapter.send_group_msg(
            group_id=self.config.report_gid, message=message
        )
