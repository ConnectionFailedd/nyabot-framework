import importlib.util
import os
import shlex
import sys
from abc import ABC, abstractmethod
from asyncio import Lock
from functools import cached_property
from inspect import Parameter, isabstract, signature
from pathlib import Path
from types import MappingProxyType, NoneType
from typing import (
    Any,
    Awaitable,
    Callable,
    Generic,
    Optional,
    Sequence,
    TypeVar,
    Union,
    cast,
    final,
    get_args,
    get_origin,
    override,
)

import nonebot
from nonebot import on_message
from nonebot.adapters.onebot.v11 import Bot, GroupMessageEvent, Message, MessageSegment
from nonebot.matcher import Matcher
from nonebot.rule import is_type
from pydantic import BaseModel, computed_field
from structlog import get_logger

from .class_property import class_property


class NyaConfig(BaseModel):
    """
    NyaPlugin 配置类
    """

    # 状态存储文件名。为 None 则不会加载/存储该插件状态。
    state_filename: Optional[str] = None

    # 插件生效的群 ID 列表，为 None 则所有群均生效
    gid_list: Optional[list[int]] = None

    # 调试群 ID，可以在该群中上报一些调试信息，不用进入后台查看。为 None 会丢弃所有上报内容。
    debug_gid: Optional[int] = None

    # 身份到 QQ 号的映射，用于限制权限
    role_to_user_id_list_map: dict[str, list[int]] = {}

    # * 自动生成
    @computed_field
    @cached_property
    def gid_set(self) -> Optional[set[int]]:
        return set(self.gid_list) if self.gid_list else None

    @computed_field
    @cached_property
    def role_to_user_id_set_map(self) -> dict[str, set[int]]:
        return {
            role: set(user_id_list)
            for role, user_id_list in self.role_to_user_id_list_map.items()
        }


class NyaState(BaseModel):
    """
    NyaPlugin 状态类。
    """


class NyaEvent:
    def __init__(self, event: GroupMessageEvent, matcher: type[Matcher]):
        self.event = event
        self.matcher = matcher

    @property
    def message(self):
        return self.event.message

    @property
    def user_id(self):
        return self.event.user_id

    @property
    def group_id(self):
        return self.event.group_id

    async def reply(self, message: str | Message | MessageSegment):
        await self.matcher.send(message)

    async def send_group_msg(
        self, message: str | Message | MessageSegment, group_id: int
    ):
        await self.matcher.send(message, group_id=group_id)


NyaConfigT = TypeVar("NyaConfigT", bound="NyaConfig")
NyaStateT = TypeVar("NyaStateT", bound="NyaState")


class NyaPlugin(Generic[NyaConfigT, NyaStateT], ABC):
    """
    群消息命令处理插件
    """

    _logger = get_logger(sender="NyaPlugin")
    _command: "RootCommand | LeafCommand"

    @class_property
    def logger(cls):
        return cls._logger

    @class_property
    def command(cls):
        return cls._command

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        # 输出 log
        NyaPlugin._logger.info("Loading plugin", plugin=cls.__name__)

        if isabstract(cls):
            NyaPlugin._logger.info("Abstract plugin, not loaded", plugin=cls.__name__)

        # 设置 _logger
        cls._logger = get_logger(sender=cls.__name__)

        # 获取 NyaConfigT 和 NyaStateT 类型
        orig_bases = cast(
            tuple[Any, ...],
            getattr(cls, "__orig_bases__", ()),
        )
        for base in orig_bases:
            origin = get_origin(base)
            if origin is NyaPlugin:
                (
                    config_cls,
                    state_cls,
                ) = get_args(base)
                cls.config_cls: type[NyaConfig] = config_cls
                cls.state_cls: type[NyaState] = state_cls
                break
        else:
            raise TypeError(
                f"{cls.__name__} must specify generic parameter NyaPlugin[NyaConfigT, NyaStateT]"
            )

        # 向 NyaFactory 注册
        NyaFactory.register(cls)

    @class_property
    def config(cls) -> NyaConfig:
        return NyaFactory.nya_config_map[cls._command.name]

    @class_property
    def state(cls) -> NyaState:
        return NyaFactory.nya_state_map[cls._command.name]

    def __init__(self, event: NyaEvent) -> None:
        self.event = event

    @final
    async def handle(self) -> None:
        self._logger.info("Called handle()", plugin=self.__class__.__name__)

        if self.config.gid_set and self.event.group_id not in self.config.gid_set:
            self._logger.info(
                "Not in limited groups",
                group_id=self.event.group_id,
                gid_set=self.config.gid_set,
            )
            return

        roles = self._get_roles()

        command_part_list = shlex.split(str(self.event.message))
        _command, raw_args, permission_denied = self._parse_command(
            command_part_list, roles
        )

        if permission_denied:
            self._logger.info(
                "Permission denied: ",
                _command=_command.full_name,
                limited_roles=_command.limited_roles,
                roles=roles,
            )
            await self.event.reply(f"{_command.full_name!r}：权限不足")
            return

        if not isinstance(_command, LeafCommand):
            # 命令不全
            if not raw_args:
                self._logger.info("Incomplete _command: ", _command=_command.full_name)
                await self.event.reply(_command.help_info(roles=roles))
                return

            # 请求帮助信息
            if raw_args[0] in ("-h", "--help"):
                await self.event.reply(_command.help_info(roles=roles))
                return

            # 错误的子命令
            self._logger.info(
                "Invalid _command: ",
                _command=_command.full_name,
                subcommand=raw_args[0],
            )
            await self.event.reply(
                f"{_command.full_name!r}：无效的子命令 {raw_args[0]!r}"
            )
            await self.event.reply(_command.help_info(roles))
            return

        # 请求帮助信息
        if raw_args and raw_args[0] in ("-h", "--help"):
            await self.event.reply(_command.help_info(roles=roles))
            return

        ret = _command.function(self, raw_args)
        if isinstance(ret, Awaitable):
            ret = await ret

        if ret.code != 0:
            if ret.log:
                self._logger.info("Error occurred: ", error=ret.log)
            else:
                self._logger.info("Unknown error occurred")
        elif ret.log:
            self._logger.info("Message: ", info=ret.log)

        if ret.reply:
            await self.event.reply(ret.reply)

        if ret.report and self.config.debug_gid:
            await self._report(message=ret.report)

        if ret.need_help:
            await self.event.reply(_command.help_info(roles))

        self._logger.info(
            "Event finished",
            event_=self.event,
            plugin=self.__class__.__name__,
        )

    @final
    async def _report(self, message: str) -> None:
        if not self.config.debug_gid:
            self._logger.info("Report group not specified, discard report info")
            return

        await self.event.send_group_msg(group_id=self.config.debug_gid, message=message)

    @final
    def _get_roles(self) -> set[str]:
        return {
            role
            for role, user_id_set in self.config.role_to_user_id_set_map.items()
            if self.event.user_id in user_id_set
        }

    @final
    @classmethod
    def _parse_command(
        cls, command_part_list: list[str], roles: set[str]
    ) -> tuple["Command", list[str], bool]:
        # 返回 <命令，参数，是否无权访问>
        if cls._command.limited_roles and not (cls._command.limited_roles & roles):
            # 命令无权限
            return cls._command, command_part_list[1:], True

        current_command = cls._command
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

    @final
    @classmethod
    async def load_state(cls) -> None:
        cls.logger.info("Loading state", plugin=cls.__class__.__name__)

        if not (state_filename := cls.config.state_filename):
            cls.logger.info("State filename not set", plugin=cls.__class__.__name__)
            return

        state_path = f"data/{state_filename}"

        if not os.path.exists(state_path):
            cls.logger.info("State file not found", plugin=cls.__class__.__name__)
            return

        with open(state_path, "r") as f:
            cls.state.model_validate_json(f.read())

        cls.logger.info("State loaded", plugin=cls.__class__.__name__)

    @final
    @classmethod
    async def save_state(cls) -> None:
        cls.logger.info("Saving state", plugin=cls.__class__.__name__)

        if not (state_filename := cls.config.state_filename):
            cls.logger.info("State filename not set", plugin=cls.__class__.__name__)
            return

        state_path = f"data/{state_filename}"

        if os.path.exists(state_path):
            cls.logger.info(
                "Original State file backed up", plugin=cls.__class__.__name__
            )
            os.rename(state_path, state_path + ".bak")

        with open(state_path, "w") as f:
            f.write(cls.state.model_dump_json(indent=4))

        cls.logger.info("State saved", plugin=cls.__class__.__name__)


class NyaFactory:
    # ! 这里的键都是 command_name
    _nya_config_map: dict[str, NyaConfig] = {}
    _nya_state_map: dict[str, NyaState] = {}
    _nya_plugin_map: dict[str, type[NyaPlugin[Any, Any]]] = {}
    _nya_plugin_instance_list_map: dict[str, list[NyaPlugin[Any, Any]]] = {}
    _nya_plugin_instance_lock: Lock = Lock()

    @class_property
    def nya_config_map(cls):
        return MappingProxyType(cls._nya_config_map)

    @class_property
    def nya_state_map(cls):
        return MappingProxyType(cls._nya_state_map)

    @class_property
    def nya_plugin_map(cls):
        return MappingProxyType(cls._nya_plugin_map)

    @class_property
    def nya_plugin_instance_list_map(cls):
        return MappingProxyType(cls._nya_plugin_instance_list_map)

    @classmethod
    def register(
        cls,
        plugin_cls: type[NyaPlugin[NyaConfigT, NyaStateT]],
    ):
        # ! 先尝试创建 NyaConfig 和 NyaState 实例，创建失败就从异常路线退出，避免注册到一半出错
        NyaPlugin.logger.info("Creating config", plugin=plugin_cls.__name__)
        config = nonebot.get_driver().config
        if config.nya_plugin_config and (
            nya_config_obj := config.nya_plugin_config.get(plugin_cls.__name__, None)
        ):
            NyaPlugin.logger.info("Config found, loading", plugin=plugin_cls.__name__)
            nya_config = plugin_cls.config_cls.model_validate(nya_config_obj)
        else:
            NyaPlugin.logger.info(
                "Config not found, creating default config", plugin=plugin_cls.__name__
            )
            nya_config = plugin_cls.config_cls()
        NyaPlugin.logger.info(
            "Config created",
            plugin=plugin_cls.__name__,
            config=nya_config.model_dump_json(),
        )

        NyaPlugin.logger.info("Creating state", plugin=plugin_cls.__name__)
        if state_filename := nya_config.state_filename:
            NyaPlugin.logger.info(
                "State filename set, try loading", plugin=plugin_cls.__name__
            )
            if not os.path.exists(f"data/{state_filename}"):
                NyaPlugin.logger.info(
                    "State file not set, creating default state",
                    plugin=plugin_cls.__name__,
                )
                nya_state = plugin_cls.state_cls()
            else:
                with open(state_filename, "r") as state_file:
                    nya_state = plugin_cls.state_cls.model_validate_json(
                        state_file.read()
                    )
        else:
            NyaPlugin.logger.info(
                "State filename not set, creating default state",
                plugin=plugin_cls.__name__,
            )
            nya_state = plugin_cls.state_cls()
        NyaPlugin.logger.info(
            "State created",
            plugin=plugin_cls.__name__,
            state=nya_state.model_dump_json(),
        )

        cls._nya_plugin_map[plugin_cls.command.name] = plugin_cls
        cls._nya_config_map[plugin_cls.command.name] = nya_config
        cls._nya_state_map[plugin_cls.command.name] = nya_state

    @classmethod
    async def allocate_plugin_instance(
        cls, command_name: str, event: GroupMessageEvent, matcher: type[Matcher]
    ) -> Optional[NyaPlugin[Any, Any]]:
        if command_name not in cls._nya_plugin_map:
            return None

        if command_name in cls._nya_plugin_instance_list_map and (
            instance_list := cls._nya_plugin_instance_list_map[command_name]
        ):
            async with cls._nya_plugin_instance_lock:
                ret = instance_list.pop()
            ret.event = NyaEvent(event=event, matcher=matcher)
            return ret

        return cls._nya_plugin_map[command_name](
            event=NyaEvent(event=event, matcher=matcher)
        )

    @classmethod
    async def release_plugin_instance(
        cls, command_name: str, plugin_instance: NyaPlugin[Any, Any]
    ):
        async with cls._nya_plugin_instance_lock:
            if command_name not in cls._nya_plugin_instance_list_map:
                cls._nya_plugin_instance_list_map[command_name] = []
            cls._nya_plugin_instance_list_map[command_name].append(plugin_instance)


class ReturnValue:
    """
    所有插件命令回调函数的返回值类型。

    :var int code: 返回状态码，0 代表成功
    :var Optional[str] stdlog: 输出到后台的 log 内容
    :var Optional[str] stdreply: 回复到当前聊天的内容
    :var Optional[str] stdreport: 推送到 bot 管理群的错误消息
    :var bool need_help: 是否需要输出帮助信息
    """

    def __init__(
        self,
        code: int,
        *,
        log: Optional[str] = None,
        reply: Optional[str] = None,
        report: Optional[str] = None,
        need_help: bool = False,
    ):
        self.code = code
        self.log = log
        self.reply = reply
        self.report = report
        self.need_help = need_help


class Function:
    """
    封装一个插件命令回调函数。
    限用于 MessageEvent 的处理函数。
    """

    @abstractmethod
    def _check(self) -> None:
        """
        检查函数描述是否符合传入的函数的签名。

        :return None: 检查通过时静默返回
        :raise ValueError: 检查不通过时抛出异常
        """
        raise NotImplementedError()

    @abstractmethod
    def __call__(
        self,
        plugin: NyaPlugin[NyaConfigT, NyaStateT],
        raw_arg_list: list[str],
    ) -> ReturnValue | Awaitable[ReturnValue]:
        """
        用命令拆分出的参数调用回调函数。

        :param Plugin[EventT, StateT, ConfigT] plugin: 插件实例
        :param list[str] raw_arg_list: 拆分出的原始参数列表
        """
        raise NotImplementedError()

    @abstractmethod
    def get_param_inline(self) -> str:
        """
        返回一行的命令参数。
        """
        raise NotImplementedError()

    @abstractmethod
    def get_param_desc(self) -> str:
        """
        返回命令参数的完整描述。
        """
        raise NotImplementedError()


class FunctionWithFixedParams(Function):
    """
    只有固定参数的插件命令回调函数。
    参数必须全部为位置参数。

    :var Callable[..., ReturnValue | Awaitable[ReturnValue]] func: 回调函数，支持同步/异步的返回值类型为 ReturnValue 的函数
    :var list[tuple[str, type, str]] fixed_params_desc: 固定参数的描述，结构为 [(参数名, 参数类型, 参数描述), ...]
    """

    class _NeverType:
        pass

    def __init__(
        self,
        func: Callable[..., ReturnValue | Awaitable[ReturnValue]],
        fixed_param_desc_list: list[tuple[str, type, str]],
    ):
        self.func = func
        self.fixed_param_desc_list = fixed_param_desc_list
        self._check()

    @override
    @final
    def _check(self) -> None:
        # 获取函数签名
        func_signature = signature(self.func)
        # 读取形参列表，去掉开头的 self 参数
        func_param_desc_list = list(func_signature.parameters.values())[1:]

        # 只允许固定参数
        for func_param_desc in func_param_desc_list:
            if func_param_desc.kind not in (
                Parameter.POSITIONAL_ONLY,
                Parameter.POSITIONAL_OR_KEYWORD,
            ):
                raise ValueError(f"Invalid parameter: {func_param_desc.name}")

        if (
            # 检查参数数量是否匹配
            len(func_param_desc_list) != len(self.fixed_param_desc_list)
            # 检查参数类型是否匹配
            or any(
                func_param_desc.annotation != fixed_param_desc[1]
                for func_param_desc, fixed_param_desc in zip(
                    func_param_desc_list, self.fixed_param_desc_list
                )
            )
        ):
            raise ValueError(
                "Invalid parameter type: "
                "expected: "
                + (
                    f"({[fixed_param_desc[1].__name__ for fixed_param_desc in self.fixed_param_desc_list]})"
                )
                + ", got: "
                + (
                    f"({[func_param_desc.annotation.__name__ for func_param_desc in func_param_desc_list]})"
                )
            )

    @override
    @final
    def __call__(
        self, plugin: NyaPlugin[NyaConfigT, NyaStateT], raw_arg_list: list[str]
    ) -> ReturnValue | Awaitable[ReturnValue]:
        # 检查参数数量
        if len(raw_arg_list) != len(self.fixed_param_desc_list):
            return ReturnValue(
                1,
                log=f"Invalid argument number: expected {len(self.fixed_param_desc_list)}, got {len(raw_arg_list)}",
                reply=f"参数数量错误：需要 {len(self.fixed_param_desc_list)} 个参数，但传入 {len(raw_arg_list)} 个参数",
                need_help=True,
            )

        # 尝试将原始参数转换到目标类型
        converted_arg_list: list[Any] = []
        for raw_fixed_arg, fixed_param_desc in zip(
            raw_arg_list, self.fixed_param_desc_list
        ):
            try:
                converted_arg_list.append(fixed_param_desc[1](raw_fixed_arg))
            except:
                return ReturnValue(
                    1,
                    log=f"Invalid argument: expected {fixed_param_desc[1].__name__}, got {raw_fixed_arg!r}",
                    reply=f"参数类型错误：参数 {fixed_param_desc[0]} 为 {fixed_param_desc[1].__name__} 类型，但传入 {raw_fixed_arg!r}",
                    need_help=True,
                )

        # 调用回调函数
        return self.func(plugin, *converted_arg_list)

    @override
    @final
    def get_param_inline(self) -> str:
        return " ".join(
            f"<{fixed_param_desc[0]}:{fixed_param_desc[1].__name__}>"
            for fixed_param_desc in self.fixed_param_desc_list
        )

    @override
    @final
    def get_param_desc(self) -> str:
        return "\n".join(
            f"* {fixed_param_desc[0]}：{fixed_param_desc[2]}"
            for fixed_param_desc in self.fixed_param_desc_list
        )


class FunctionWithOptionalParam(Function):
    """
    带一个可选参数的插件命令回调函数。
    参数必须全部为位置参数，可选参数需放在函数参数列表最后。

    :var Callable[..., ReturnValue | Awaitable[ReturnValue]] func: 回调函数，支持同步/异步的返回值类型为 ReturnValue 的函数
    :var list[tuple[str, type, str]] fixed_params_desc: 固定参数的描述，结构为 [(参数名, 参数类型, 参数描述), ...]
    :var tuple[str, type, str] optional_param_desc: 可选参数的描述，结构为 (参数名, 参数类型, 参数描述)
    """

    def __init__(
        self,
        func: Callable[..., ReturnValue | Awaitable[ReturnValue]],
        fixed_param_desc_list: list[tuple[str, type, str]],
        optional_param_desc: tuple[str, type, str],
    ):
        self.func = func
        self.fixed_param_desc_list = fixed_param_desc_list
        self.optional_param_desc = optional_param_desc
        self._check()

    def _check(self) -> None:
        # 获取函数签名
        func_signature = signature(self.func)
        # 读取形参列表，去掉开头的 self 参数
        func_param_desc_list = list(func_signature.parameters.values())[1:]

        # 只接受固定参数
        for func_param_desc in func_param_desc_list:
            if func_param_desc.kind not in (
                Parameter.POSITIONAL_ONLY,
                Parameter.POSITIONAL_OR_KEYWORD,
            ):
                raise ValueError(f"Invalid parameter: {func_param_desc.name}")

        if (
            # 检查参数数量是否正确
            len(func_param_desc_list) != len(self.fixed_param_desc_list) + 1
            # 检查固定参数类型是否匹配
            or any(
                func_param_desc.annotation != fixed_param_desc[1]
                for func_param_desc, fixed_param_desc in zip(
                    func_param_desc_list[:-1], self.fixed_param_desc_list
                )
            )
            # 检查可选参数类型是否匹配
            or func_param_desc_list[-1].annotation
            != Optional[self.optional_param_desc[1]]
        ):
            raise ValueError(
                "Invalid parameter type: "
                "expected: "
                + (
                    f"({[fixed_param_desc[1].__name__ for fixed_param_desc in self.fixed_param_desc_list]}, Optional[{self.optional_param_desc[1].__name__}])"
                )
                + ", got: "
                + (
                    f"({[func_param_desc.annotation.__name__ for func_param_desc in func_param_desc_list]})"
                    # 最后一个参数不是 Optional
                    if get_origin(func_param_desc_list[-1].annotation) is not Union
                    or NoneType not in get_args(func_param_desc_list[-1].annotation)
                    else (
                        f"({[func_param_desc.annotation.__name__ for func_param_desc in func_param_desc_list[:-1]]}, "
                        + f"{" | ".join([type_.__name__ for type_ in get_args(func_param_desc_list[-1].annotation) if type_ is not NoneType])} ...)"
                    )
                )
            )

    def __call__(
        self,
        plugin: NyaPlugin[NyaConfigT, NyaStateT],
        raw_arg_list: list[str],
    ) -> ReturnValue | Awaitable[ReturnValue]:
        # 检查参数数量
        if len(raw_arg_list) < len(self.fixed_param_desc_list):
            return ReturnValue(
                1,
                log=f"Invalid argument number: expected {len(self.fixed_param_desc_list)} or {len(self.fixed_param_desc_list) + 1}, got {len(raw_arg_list)}",
                reply=f"参数数量错误：需要 {len(self.fixed_param_desc_list)} 或 {len(self.fixed_param_desc_list) + 1} 个参数，但传入 {len(raw_arg_list)} 个参数",
                need_help=True,
            )

        # 尝试将原始参数转换到目标类型
        converted_arg_list: list[Any] = []
        for raw_fixed_arg, fixed_param_desc in zip(
            raw_arg_list[: len(self.fixed_param_desc_list)], self.fixed_param_desc_list
        ):
            try:
                converted_arg_list.append(fixed_param_desc[1](raw_fixed_arg))
            except:
                return ReturnValue(
                    1,
                    log=f"Invalid argument: expected {fixed_param_desc[1].__name__}, got {raw_fixed_arg!r}",
                    reply=f"参数类型错误：参数 {fixed_param_desc[0]} 为 {fixed_param_desc[1].__name__} 类型，但传入 {raw_fixed_arg!r}",
                    need_help=True,
                )
        if len(raw_arg_list) > len(self.fixed_param_desc_list):
            try:
                converted_arg_list.append(self.optional_param_desc[1](raw_arg_list[-1]))
            except:
                return ReturnValue(
                    1,
                    log=f"Invalid argument: expected Optional[{self.optional_param_desc[1].__name__}], got {raw_arg_list[-1]!r}",
                    reply=f"参数类型错误：参数 {self.optional_param_desc[0]} 为 Optional[{self.optional_param_desc[1].__name__}] 类型，但传入 {raw_arg_list[-1]!r}",
                    need_help=True,
                )
        else:
            converted_arg_list.append(None)

        # 调用回调函数
        return self.func(plugin, *converted_arg_list)

    @override
    @final
    def get_param_inline(self) -> str:
        return (
            " ".join(
                f"<{fixed_param_desc[0]}:{fixed_param_desc[1].__name__}>"
                for fixed_param_desc in self.fixed_param_desc_list
            )
            + f"[<{self.optional_param_desc[0]}:{self.optional_param_desc[1].__name__}>]"
        )

    @override
    @final
    def get_param_desc(self) -> str:
        fixed_param_desc_str = "\n".join(
            f"* {fixed_param_desc[0]}：{fixed_param_desc[2]}"
            for fixed_param_desc in self.fixed_param_desc_list
        )
        optional_param_desc_str = (
            f"* {self.optional_param_desc[0]}：{self.optional_param_desc[2]}"
        )
        return (
            f"{fixed_param_desc_str}\n{optional_param_desc_str}"
            if fixed_param_desc_str
            else optional_param_desc_str
        )


class FunctionWithVariableParams(Function):
    """
    带一个可变参数的插件命令回调函数。
    参数必须全部为位置参数，可变参数需放在函数参数列表最后。

    :var Callable[..., ReturnValue | Awaitable[ReturnValue]] func: 回调函数，支持同步/异步的返回值类型为 ReturnValue 的函数
    :var list[tuple[str, type, str]] fixed_params_desc: 固定参数的描述，结构为 [(参数名, 参数类型, 参数描述), ...]
    :var tuple[str, type, str] variable_param_desc: 可变参数的描述，结构为 (参数名, 参数类型, 参数描述)
    """

    def __init__(
        self,
        func: Callable[..., ReturnValue | Awaitable[ReturnValue]],
        fixed_param_desc_list: list[tuple[str, type, str]],
        variable_param_desc: tuple[str, type, str],
    ):
        self.func = func
        self.fixed_param_desc_list = fixed_param_desc_list
        self.variable_param_desc = variable_param_desc
        self._check()

    def _check(self) -> None:
        # 获取函数签名
        func_signature = signature(self.func)
        # 读取形参列表，去掉开头的 self 参数
        func_param_desc_list = list(func_signature.parameters.values())[1:]

        # 只接受位置参数
        for func_param_desc in func_param_desc_list:
            if func_param_desc.kind not in (
                Parameter.POSITIONAL_ONLY,
                Parameter.POSITIONAL_OR_KEYWORD,
                Parameter.VAR_POSITIONAL,
            ):
                raise ValueError(f"Invalid parameter: {func_param_desc.name}")

        if (
            # 检查参数数量是否正确
            len(func_param_desc_list) != len(self.fixed_param_desc_list) + 1
            # 检查最后一个参数是否为可变参数
            or func_param_desc_list[-1].kind != Parameter.VAR_POSITIONAL
            # 检查固定参数类型是否匹配
            or any(
                func_param_desc.annotation != fixed_param_desc[1]
                for func_param_desc, fixed_param_desc in zip(
                    func_param_desc_list[:-1], self.fixed_param_desc_list
                )
            )
            # 检查可变参数类型是否匹配
            or func_param_desc_list[-1].annotation != self.variable_param_desc[1]
        ):
            raise ValueError(
                "Invalid parameter type: "
                "expected: "
                + (
                    f"({[fixed_param_desc[1].__name__ for fixed_param_desc in self.fixed_param_desc_list]}, {self.variable_param_desc[1].__name__} ...)"
                )
                + ", got: "
                + (
                    f"({[func_param_desc.annotation.__name__ for func_param_desc in func_param_desc_list]})"
                    if func_param_desc_list[-1].kind != Parameter.VAR_POSITIONAL
                    else f"({[func_param_desc.annotation.__name__ for func_param_desc in func_param_desc_list[:-1]]}, {func_param_desc_list[-1].annotation.__name__} ...)"
                )
            )

    def __call__(
        self,
        plugin: NyaPlugin[NyaConfigT, NyaStateT],
        raw_arg_list: list[str],
    ) -> ReturnValue | Awaitable[ReturnValue]:
        # 检查参数数量
        if len(raw_arg_list) < len(self.fixed_param_desc_list):
            return ReturnValue(
                1,
                log=f"Invalid argument number: expected at least {len(self.fixed_param_desc_list)}, got {len(raw_arg_list)}",
                reply=f"参数数量错误：需要至少 {len(self.fixed_param_desc_list)} 个参数，但传入 {len(raw_arg_list)} 个参数",
                need_help=True,
            )

        # 尝试将原始参数转换到目标类型
        converted_arg_list: list[Any] = []
        for raw_fixed_arg, fixed_param_desc in zip(
            raw_arg_list[: len(self.fixed_param_desc_list)], self.fixed_param_desc_list
        ):
            try:
                converted_arg_list.append(fixed_param_desc[1](raw_fixed_arg))
            except:
                return ReturnValue(
                    1,
                    log=f"Invalid argument: expected {fixed_param_desc[1].__name__}, got {raw_fixed_arg!r}",
                    reply=f"参数类型错误：参数 {fixed_param_desc[0]} 为 {fixed_param_desc[1].__name__} 类型，但传入 {raw_fixed_arg!r}",
                    need_help=True,
                )
        for raw_variable_arg in raw_arg_list[len(self.fixed_param_desc_list) :]:
            try:
                converted_arg_list.append(self.variable_param_desc[1](raw_variable_arg))
            except:
                return ReturnValue(
                    1,
                    log=f"Invalid argument: expected {self.variable_param_desc[1].__name__}, got {raw_variable_arg!r}",
                    reply=f"参数类型错误：参数 {self.variable_param_desc[0]} 为 {self.variable_param_desc[1].__name__} 类型，但传入 {raw_variable_arg!r}",
                    need_help=True,
                )

        # 调用回调函数
        return self.func(plugin, *converted_arg_list)

    @override
    @final
    def get_param_inline(self) -> str:
        return (
            " ".join(
                f"<{fixed_param_desc[0]}:{fixed_param_desc[1].__name__}>"
                for fixed_param_desc in self.fixed_param_desc_list
            )
            + f"[<{self.variable_param_desc[0]}:{self.variable_param_desc[1].__name__}> ...]"
        )

    @override
    @final
    def get_param_desc(self) -> str:
        fixed_param_desc_str = "\n".join(
            f"* {fixed_param_desc[0]}：{fixed_param_desc[2]}"
            for fixed_param_desc in self.fixed_param_desc_list
        )
        variable_param_desc_str = (
            f"* {self.variable_param_desc[0]}：{self.variable_param_desc[2]}"
        )
        return (
            f"{fixed_param_desc_str}\n{variable_param_desc_str}"
            if fixed_param_desc_str
            else variable_param_desc_str
        )


class Command:
    """
    描述一个插件的处理的命令的基类。

    :var str name: 命令名称
    :var str desc: 命令描述
    :var Optional[set[str]] limited_roles: 限定身份，为 None 表示所有人可访问
    """

    def __init__(self, name: str, desc: str, limited_roles: Optional[set[str]]):
        self.name = name
        self.desc = desc
        self.limited_roles = limited_roles

        # 需要在 Command 基类中有这个成员
        self.full_name = name

    @abstractmethod
    def help_info(self, roles: set[str]) -> str:
        """
        输出命令的帮助信息

        :param set[str] roles: 消息发送者身份
        """
        raise NotImplementedError()

    @abstractmethod
    def _update_full_name(self, parent_full_name: str) -> None:
        """
        更新命令的全称。

        :param str parent_full_name: 父命令的全称
        """
        raise NotImplementedError()


class InternalCommand(Command):
    """
    中间命令。
    命令树的中间节点，有若干子命令。

    :var str name: 命令名称
    :var str desc: 命令描述
    :var Optional[set[str]] limited_roles: 限定身份，为 None 表示所有人可访问
    :var list[Command] subcommand_list: 子命令列表
    """

    def __init__(
        self,
        name: str,
        desc: str,
        limited_roles: Optional[set[str]],
        subcommand_list: Sequence[Command],
    ):
        super().__init__(name, desc, limited_roles)
        self.subcommand_list = subcommand_list

    @final
    def help_info(self, roles: set[str]) -> str:
        # 执行到这里说明用户有该命令的权限
        assert (self.limited_roles is None) or (roles & self.limited_roles)

        # 应该假设用户至少有一条子命令的权限
        return (
            f"{self.full_name}\n"
            + f"{self.desc}\n"
            + "子命令列表：\n"
            + "\n".join(
                f"* {subcommand.name}: {subcommand.desc}"
                for subcommand in self.subcommand_list
                if (
                    (subcommand.limited_roles is None)
                    or (roles & subcommand.limited_roles)
                )
            )
        )

    @final
    def _update_full_name(self, parent_full_name: str) -> None:
        self.full_name = f"{parent_full_name} {self.name}"
        for subcommand in self.subcommand_list:
            subcommand._update_full_name(self.full_name)


class RootCommand(InternalCommand):
    """
    根命令。
    命令树的根节点，有若干子命令。和中间命令的区别是，在初始化完成后会主动向下更新所有命令的 full_name。
    如果命令树只有一层，请使用叶子命令。

    :var str name: 命令名称
    :var str desc: 命令描述
    :var Optional[set[str]] limited_roles: 限定身份，为 None 表示所有人可访问
    :var list[Command] subcommand_list: 子命令列表
    """

    def __init__(
        self,
        name: str,
        desc: str,
        limited_roles: Optional[set[str]],
        subcommand_list: Sequence[Command],
    ):
        super().__init__(name, desc, limited_roles, subcommand_list)

        # 主动更新孩子节点的 full_name
        for subcommand in self.subcommand_list:
            subcommand._update_full_name(self.name)


class LeafCommand(Command):
    """
    叶子命令。
    命令树的叶子节点，保存一个命令回调函数。

    :var str name: 命令名称
    :var str desc: 命令描述
    :var Optional[set[str]] limited_roles: 限定身份，为 None 表示所有人可访问
    :var list[Command] subcommand_list: 子命令列表
    """

    def __init__(
        self,
        name: str,
        desc: str,
        limited_roles: Optional[set[str]],
        function: Function,
    ):
        super().__init__(name, desc, limited_roles)
        self.function = function

    @abstractmethod
    def help_info(self, roles: set[str]) -> str:
        """
        输出命令的帮助信息

        :param set[str] roles: 消息发送者身份
        """
        # 执行到这里说明用户有该命令的权限
        assert (self.limited_roles is None) or (roles & self.limited_roles)

        return f"{self.full_name} {self.function.get_param_inline()}\n" + (
            f"参数列表：\n{param_desc}"
            if (param_desc := self.function.get_param_desc())
            else "本命令没有参数"
        )

    @abstractmethod
    def _update_full_name(self, parent_full_name: str) -> None:
        """
        更新命令的全称。

        :param str parent_full_name: 父命令的全称
        """
        self.full_name = f"{parent_full_name} {self.name}"


group_message_matcher = on_message(rule=is_type(GroupMessageEvent))


@group_message_matcher.handle()
async def command_dispatch(event: GroupMessageEvent):
    command_name = str(event.message).split()[0]

    NyaPlugin.logger.info("Called command_dispatch", command_name=command_name)

    plugin_instance = await NyaFactory.allocate_plugin_instance(
        command_name, event, group_message_matcher
    )

    if plugin_instance:
        NyaPlugin.logger.info(
            "Plugin matched, processing",
            command_name=command_name,
            plugin=plugin_instance.__class__.__name__,
        )
        await plugin_instance.handle()
        await NyaFactory.release_plugin_instance(command_name, plugin_instance)
    else:
        NyaPlugin.logger.info(
            "No plugin matched, discarded", command_name=command_name
        )

    await group_message_matcher.finish()


def load_nya_plugin(path: Path):
    NyaPlugin.logger.info("Loading plugin", path=path)

    if path.is_file() and path.suffix == ".py":
        NyaPlugin.logger.info("Loading plugin from file", file=path)

        spec = importlib.util.spec_from_file_location(path.stem, path)  # 模块名
    elif path.is_dir() and (init := (path / "__init__.py")).exists() and init.is_file():
        NyaPlugin.logger.info("Loading plugin from directory", dir=path)

        spec = importlib.util.spec_from_file_location(
            path.name, init, submodule_search_locations=[str(path)]
        )
    else:
        NyaPlugin.logger.info("Invalid path type, skipped", path=path)
        return

    assert spec and spec.loader

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)


def load_nya_plugins(nya_plugin_dir: str | Path):
    nya_plugin_dir = Path(nya_plugin_dir)

    NyaPlugin.logger.info("Loading plugins", dir=nya_plugin_dir)

    for nya_plugin in nya_plugin_dir.iterdir():
        load_nya_plugin(nya_plugin)

    NyaPlugin.logger.info("All plugins loaded", dir=nya_plugin_dir)
