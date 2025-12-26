import nonebot
from nonebot.adapters.onebot.v11 import Adapter

# 初始化 NoneBot
nonebot.init()

# 注册适配器
driver = nonebot.get_driver()
driver.register_adapter(Adapter)

# 加载 NyaPlugin
nonebot.load_plugin("nya_plugin")  # 本地插件

config = driver.config
if nya_plugin_dir := config.nya_plugin_dir:
    from nya_plugin import load_nya_plugins

    load_nya_plugins(nya_plugin_dir)

if __name__ == "__main__":
    nonebot.run()
