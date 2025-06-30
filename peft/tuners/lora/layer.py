import os
import importlib

# 根据环境变量选择 layer 类型
layer_type = os.getenv("PEFT_LAYER_TYPE", "lora")
module_name = f"peft.tuners.lora.{layer_type}layer"

# 动态导入
layer_module = importlib.import_module(module_name)

# 将导入模块的所有“非私有成员”放进当前 module 作用域
for attr in dir(layer_module):
    globals()[attr] = getattr(layer_module, attr)