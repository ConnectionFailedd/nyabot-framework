import shlex
from abc import ABC
from functools import cached_property
from typing import Any, Awaitable, Optional, TypeVar, final, override

from alicebot.adapter.cqhttp.event import GroupMessageEvent
from pydantic import computed_field
from structlog import get_logger

from nyaplugin.nyaplugin_base import NYAPlugin, NYAPluginConfig, NYAPluginState

from .command import Command, InternalCommand, LeafCommand, RootCommand

EventT = TypeVar("EventT", bound=GroupMessageEvent)
StateT = TypeVar("StateT", bound="NYAPluginState")
ConfigT = TypeVar("ConfigT", bound="CQHTTPGroupMessageCommandHandlerPluginConfig")


class CQHTTPGroupMessageCommandHandlerPluginConfig(NYAPluginConfig):
    # 限制群 ID，值为 None 代表所有群均可用
    limited_group_id_list: Optional[list[int]] = None

    # 身份到 QQ 号的映射，用于限制权限
    role_to_qq_uin_list_map: dict[str, list[int]] = {}

    # 自动生成
    # ! 使用 model_validator 会有一个问题：上面两个字段可能也会在 model_validator 中提供，导致下面的字段为空
    @computed_field
    @cached_property
    def limited_group_id_set(self) -> Optional[set[int]]:
        return set(self.limited_group_id_list) if self.limited_group_id_list else None

    @computed_field
    @cached_property
    def role_to_qq_uin_set_map(self) -> dict[str, set[int]]:
        return {
            role: set(qq_uin_list)
            for role, qq_uin_list in self.role_to_qq_uin_list_map.items()
        }


class CQHTTPGroupMessageCommandHandlerPlugin(NYAPlugin[EventT, StateT, ConfigT], ABC):
    """
    CQHTTP 接口上的群消息命令处理插件。
    """

    logger = get_logger(plugin="CQHTTPGroupMessageCommandHandlerPlugin")
    command: RootCommand | LeafCommand

    def __init_subclass__(
        cls,
        config: type[ConfigT] | None = None,
        init_state: StateT | None = None,
        **_kwargs: Any,
    ) -> None:
        super().__init_subclass__(config, init_state, **_kwargs)

        cls.logger = get_logger(plugin=cls.__name__)

        CQHTTPGroupMessageCommandHandlerPlugin.logger.info(
            "Loading plugin: ",
            base=CQHTTPGroupMessageCommandHandlerPlugin.__name__,
            plugin=cls.__name__,
        )

        if not hasattr(cls, "command") or not isinstance(
            getattr(cls, "command"), RootCommand | LeafCommand
        ):
            raise ValueError(
                f'Missing required attribute "command: RootCommand | LeafCommand" in plugin {cls.__name__!r}'
            )

        cls.registered_nyaplugin_list.append(cls)

    @override
    @final
    async def rule(self) -> bool:
        self.logger.info(
            f"Received Event: ",
            event_=self.event,
            plugin=self.__class__.__name__,
        )

        # 运行时 event 类型仍可能不满足，需检查
        if not isinstance(self.event, GroupMessageEvent):  # type: ignore
            self.logger.info(f"Event rejected: Not a GroupMessageEvent")
            return False

        # 群聊不在限定的群聊中，静默丢弃
        if (limited_group_id_set := self.config.limited_group_id_set) and (
            group_id := self.event.group_id
        ) not in limited_group_id_set:
            self.logger.info(
                f"Event rejected: Filtered by limited_group_ids: ",
                group_id=group_id,
                limited_group_id_set=limited_group_id_set,
            )
            return False

        # 消息的第一部分不是本命令名，静默丢弃
        if not (first_command_part := str(self.event.message).strip().split()[0]) == (
            command_name := self.__class__.command.name
        ):
            self.logger.info(
                f"Event rejected: Filtered by command name: ",
                first_command_part=first_command_part,
                command_name=command_name,
            )
            return False

        self.logger.info(f"Event accepted")
        return True

    @override
    @final
    async def handle(self) -> None:
        self.logger.info(
            "Event handling",
            event_=self.event,
            plugin=self.__class__.__name__,
        )

        roles = self._get_roles()

        command_part_list = shlex.split(str(self.event.message))
        command, raw_args, permission_denied = self._parse_command(
            command_part_list, roles
        )

        if permission_denied:
            self.logger.error(
                "Permission denied: ",
                command=command.full_name,
                limited_roles=command.limited_roles,
                roles=roles,
            )
            await self.event.reply(f"{command.full_name!r}：权限不足")
            return

        if not isinstance(command, LeafCommand):
            # 命令不全
            if not raw_args:
                self.logger.error("Incomplete command: ", command=command.full_name)
                await self.event.reply(command.help_info(roles=roles))
                return

            # 请求帮助信息
            if raw_args[0] in ("-h", "--help"):
                await self.event.reply(command.help_info(roles=roles))
                return

            # 错误的子命令
            self.logger.error(
                "Invalid command: ", command=command.full_name, subcommand=raw_args[0]
            )
            await self.event.reply(
                f"{command.full_name!r}：无效的子命令 {raw_args[0]!r}"
            )
            await self.event.reply(command.help_info(roles))
            return

        # 请求帮助信息
        if raw_args and raw_args[0] in ("-h", "--help"):
            await self.event.reply(command.help_info(roles=roles))
            return

        ret = command.function(self, raw_args)
        if isinstance(ret, Awaitable):
            ret = await ret

        if ret.code != 0:
            if ret.log:
                self.logger.error("Error occurred: ", error=ret.log)
            else:
                self.logger.error("Unknown error occurred")
        elif ret.log:
            self.logger.info("Message: ", info=ret.log)

        if ret.reply:
            await self.event.reply(ret.reply)

        if ret.report and self.config.report_gid:
            await self.report(message=ret.report)

        if ret.need_help:
            await self.event.reply(command.help_info(roles))

        self.logger.info(
            "Event finished",
            event_=self.event,
            plugin=self.__class__.__name__,
        )

    @final
    def _get_roles(self) -> set[str]:
        return {
            role
            for role, qq_uin_set in self.config.role_to_qq_uin_set_map.items()
            if self.event.sender.user_id in qq_uin_set
        }

    @final
    def _parse_command(
        self, command_part_list: list[str], roles: set[str]
    ) -> tuple[Command, list[str], bool]:
        # 返回 <命令，参数，是否无权访问>
        if self.command.limited_roles and not (self.command.limited_roles & roles):
            # 命令无权限
            return self.command, command_part_list[1:], True

        current_command = self.command
        subcommand_idx = 1
        while isinstance(current_command, InternalCommand):
            # 命令参数消耗完
            if subcommand_idx == len(command_part_list):
                return current_command, [], False

            # 查找子命令
            next_command = next(
                (
                    subcommand
                    for subcommand in current_command.subcommand_list
                    if subcommand.name == command_part_list[subcommand_idx]
                ),
                None,
            )

            # 未找到子命令
            if next_command is None:
                return current_command, command_part_list[subcommand_idx:], False
            # 子命令无权限
            if next_command.limited_roles and not (next_command.limited_roles & roles):
                return next_command, command_part_list[subcommand_idx + 1 :], True

            current_command = next_command
            subcommand_idx += 1

        # 返回找到的子命令和参数
        return current_command, command_part_list[subcommand_idx:], False
