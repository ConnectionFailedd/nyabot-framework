from typing import TypeVar

from alicebot.adapter.cqhttp.event import GroupMessageEvent

from nyaplugin import NYAPluginState
from nyaplugin.command_handler import (
    CQHTTPGroupMessageCommandHandlerPlugin,
    CQHTTPGroupMessageCommandHandlerPluginConfig,
    FunctionWithVariableParams,
    LeafCommand,
    ReturnValue,
)

EventT = TypeVar("EventT", bound=GroupMessageEvent)
StateT = TypeVar("StateT")
ConfigT = TypeVar("ConfigT", bound="CQHTTPGroupMessageCommandHandlerPluginConfig")


class CommandHelperPluginConfig(CQHTTPGroupMessageCommandHandlerPluginConfig):
    __config_name__ = "command_helper"


class CommandHelperPlugin(
    CQHTTPGroupMessageCommandHandlerPlugin[
        GroupMessageEvent, NYAPluginState, CommandHelperPluginConfig
    ]
):
    """
    'help': NYABot 帮助命令
    """

    def _get_available_command_handler_plugin_list(
        self,
    ) -> list[
        type[
            CQHTTPGroupMessageCommandHandlerPlugin[
                GroupMessageEvent,
                NYAPluginState,
                CQHTTPGroupMessageCommandHandlerPluginConfig,
            ]
        ]
    ]:
        available_command_handler_plugin_list: list[
            type[
                CQHTTPGroupMessageCommandHandlerPlugin[
                    GroupMessageEvent,
                    NYAPluginState,
                    CQHTTPGroupMessageCommandHandlerPluginConfig,
                ]
            ]
        ] = []

        for plugin in self.registered_nyaplugin_list:
            # 筛选命令处理插件
            if not issubclass(plugin, CQHTTPGroupMessageCommandHandlerPlugin):
                continue

            plugin_instance = plugin()
            plugin_instance.event = self.event
            # 根据群 ID 筛选
            if (
                plugin_instance.config.limited_group_id_set
                and self.event.group_id
                not in plugin_instance.config.limited_group_id_set
            ):
                continue

            # 根据身份筛选
            if plugin.command.limited_roles and not (
                plugin.command.limited_roles & plugin_instance._get_roles()
            ):
                continue

            available_command_handler_plugin_list.append(plugin)

        return available_command_handler_plugin_list

    async def help(self, *command_parts: str) -> ReturnValue:
        # 没有参数，列出所有可用命令
        if not command_parts:
            available_command_handler_plugin_list = (
                self._get_available_command_handler_plugin_list()
            )
            await self.event.reply(
                (
                    "可用的命令列表：\n"
                    + "\n".join(
                        f"* {registered_plugin.command.name}：{registered_plugin.command.desc}"
                        for registered_plugin in available_command_handler_plugin_list
                    )
                )
                if available_command_handler_plugin_list
                else "没有可用命令"
            )
            return ReturnValue(0)

        # 查找对应的命令处理插件
        plugin = next(
            (
                plugin
                for plugin in self.registered_nyaplugin_list
                if issubclass(plugin, CQHTTPGroupMessageCommandHandlerPlugin)
                and plugin.command.name == command_parts[0]
            ),
            None,
        )
        if not plugin:
            await self.event.reply(f"找不到命令：{command_parts[0]}")
            available_command_handler_plugin_list = (
                self._get_available_command_handler_plugin_list()
            )
            await self.event.reply(
                (
                    "可用的命令列表：\n"
                    + "\n".join(
                        f"* {registered_plugin.command.name}：{registered_plugin.command.desc}"
                        for registered_plugin in available_command_handler_plugin_list
                    )
                )
                if available_command_handler_plugin_list
                else "没有可用命令"
            )
            return ReturnValue(1, log=f"Command not found: {command_parts[0]}")

        # 查找对应的命令
        plugin_instance = plugin()
        plugin_instance.event = self.event
        roles = plugin_instance._get_roles()
        command, args, permission_denied = plugin_instance._parse_command(
            list(command_parts), roles
        )

        if permission_denied:
            await self.event.reply(f"命令权限不足：{command.full_name!r}")
            return ReturnValue(1, log=f"Permission denied: {command.full_name}")

        if args:
            await self.event.reply(f"无效命令：{command.full_name +" "+ args[0]!r}")
            return ReturnValue(
                1, log=f"Invalid command: {command.full_name + " " + args[0]!r}"
            )

        await self.event.reply(command.help_info(roles))
        return ReturnValue(0)

    command = LeafCommand(
        name="help",
        desc="NYABot 帮助命令",
        limited_roles=None,  # 帮助指令对所有人开放
        function=FunctionWithVariableParams(
            func=help,
            fixed_param_desc_list=[],
            variable_param_desc=(
                "command",
                str,
                "需要查看帮助的命令，为空会列出所有可用命令",
            ),
        ),
    )
