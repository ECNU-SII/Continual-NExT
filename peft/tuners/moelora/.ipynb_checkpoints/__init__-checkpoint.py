# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from peft.import_utils import is_bnb_4bit_available, is_bnb_available, is_eetq_available
from peft.utils import register_peft_method

from .config import EvaConfig, LoftQConfig, MoELoraConfig, LoraRuntimeConfig
from .eva import get_eva_state_dict, initialize_lora_eva_weights
from .gptq import GPTQLoraLinear
from .layer import Conv2d, Conv3d, Embedding, Linear, LoraLayer
from .model import MoELoraModel

__all__ = [
    "Conv2d",
    "Conv3d",
    "Embedding",
    "EvaConfig",
    "GPTQLoraLinear",
    "Linear",
    "LoftQConfig",
    "MoELoraConfig",
    "LoraLayer",
    "MoELoraModel",
    "LoraRuntimeConfig",
    "get_eva_state_dict",
    "initialize_lora_eva_weights",
]

register_peft_method(name="moelora", config_cls=MoELoraConfig, model_cls=MoELoraModel, is_mixed_compatible=True)


def __getattr__(name):
    if (name == "Linear8bitLt") and is_bnb_available():
        from .bnb import Linear8bitLt

        return Linear8bitLt

    if (name == "Linear4bit") and is_bnb_4bit_available():
        from .bnb import Linear4bit

        return Linear4bit

    if (name == "EetqLoraLinear") and is_eetq_available():
        from .eetq import EetqLoraLinear

        return EetqLoraLinear

    raise AttributeError(f"module {__name__} has no attribute {name}")
