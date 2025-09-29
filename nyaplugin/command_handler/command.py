import inspect
from abc import abstractmethod
from types import NoneType
from typing import (
    Any,
    Awaitable,
    Callable,
    Optional,
    Sequence,
    Union,
    final,
    get_args,
    get_origin,
    override,
)

from alicebot import Plugin
from alicebot.typing import ConfigT, EventT, StateT


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
        plugin: Plugin[EventT, StateT, ConfigT],
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
        func_signature = inspect.signature(self.func)
        # 读取形参列表，去掉开头的 self 参数
        func_param_desc_list = list(func_signature.parameters.values())[1:]

        # 只允许固定参数
        for func_param_desc in func_param_desc_list:
            if func_param_desc.kind not in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
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
        self, plugin: Plugin[EventT, StateT, ConfigT], raw_arg_list: list[str]
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
        func_signature = inspect.signature(self.func)
        # 读取形参列表，去掉开头的 self 参数
        func_param_desc_list = list(func_signature.parameters.values())[1:]

        # 只接受固定参数
        for func_param_desc in func_param_desc_list:
            if func_param_desc.kind not in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
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
        plugin: Plugin[EventT, StateT, ConfigT],
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
        func_signature = inspect.signature(self.func)
        # 读取形参列表，去掉开头的 self 参数
        func_param_desc_list = list(func_signature.parameters.values())[1:]

        # 只接受位置参数
        for func_param_desc in func_param_desc_list:
            if func_param_desc.kind not in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL,
            ):
                raise ValueError(f"Invalid parameter: {func_param_desc.name}")

        if (
            # 检查参数数量是否正确
            len(func_param_desc_list) != len(self.fixed_param_desc_list) + 1
            # 检查最后一个参数是否为可变参数
            or func_param_desc_list[-1].kind != inspect.Parameter.VAR_POSITIONAL
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
                    if func_param_desc_list[-1].kind != inspect.Parameter.VAR_POSITIONAL
                    else f"({[func_param_desc.annotation.__name__ for func_param_desc in func_param_desc_list[:-1]]}, {func_param_desc_list[-1].annotation.__name__} ...)"
                )
            )

    def __call__(
        self,
        plugin: Plugin[EventT, StateT, ConfigT],
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


# TODO: test 1) report, 2) optional parameter, 3) auto load/save state
# class Command:
#     """
#     描述一个插件的所有命令。
#     命令的结构为树形，支持子命令。
#     限用于 MessageEvent 的处理函数。

#     :var str name: 命令名称
#     :var str full_name: 完整命令名
#     :var str desc: 命令描述
#     :var list[Command] subcommands: 子命令列表，与命令处理函数互斥
#     :var Function function: 命令处理函数，与子命令互斥
#     :var Optional[set[str]] limited_roles: 限定身份，为 None 表示所有人可访问
#     """

#     def __init__(
#         self,
#         name: str,
#         full_name: str,
#         desc: str,
#         subcommands_or_function: list["Command"] | Function,
#         limited_roles: Optional[set[str]],
#     ):
#         self.name: str = name
#         self.full_name: str = full_name
#         self.desc: str = desc
#         if isinstance(subcommands_or_function, list):
#             self.subcommands = subcommands_or_function
#             self.function = None
#         else:
#             self.subcommands = None
#             self.function = subcommands_or_function
#         self.limited_roles = limited_roles

#     @final
#     def help_info(self, roles: set[str]) -> str:
#         """
#         输出命令的帮助信息

#         :param roles: 消息发送者身份
#         :type roles: set[str]
#         """
#         # 执行到这里说明用户有该命令的权限
#         assert (self.limited_roles is None) or (roles & self.limited_roles)

#         # 如果命令有子命令，那么应该假设用户至少有一条子命令的权限
#         if self.subcommands:
#             return (
#                 f"{self.full_name}\n"
#                 + f"{self.desc}\n"
#                 + "子命令列表：\n"
#                 + "\n".join(
#                     f"* {subcommand.name}: {subcommand.desc}"
#                     for subcommand in self.subcommands
#                     if (
#                         (subcommand.limited_roles is None)
#                         or (roles & subcommand.limited_roles)
#                     )
#                 )
#             )
#         # 否则打印该命令的用法
#         else:
#             assert self.function
#             return f"{self.full_name} {self.function.get_param_inline()}\n" + (
#                 f"参数列表：\n{param_desc}"
#                 if (param_desc := self.function.get_param_desc())
#                 else "本命令没有参数"
#             )
