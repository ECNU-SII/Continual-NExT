# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from huggingface/transformers.
import inspect
import os
from contextlib import contextmanager, nullcontext
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from peft import PeftModel
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import EvalPrediction
from transformers import Seq2SeqTrainer as HfSeq2SeqTrainer
from transformers import Trainer as HfTrainer
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.utils import is_peft_available
from swift.llm.train.distributed_utils import (
    is_distributed, get_rank, get_world_size, is_main_process,
    get_deepspeed_zero_stage, gather_parameters, synchronize_gradients,
    all_reduce_tensor, broadcast_object
)
from swift.utils import JsonlWriter, Serializer, gc_collect, get_logger, is_mp
from .arguments import Seq2SeqTrainingArguments, TrainingArguments
from .mixin import DataLoaderMixin, SwiftMixin
import datasets
from torch.utils.data import DataLoader

import contextlib
import copy
import functools
import glob
import importlib.metadata
import inspect
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
import warnings
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.hyperparameter_search import ALL_HYPERPARAMETER_SEARCH_BACKENDS, default_hp_search_backend
from transformers.image_processing_utils import BaseImageProcessor
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from transformers.integrations.tpu import tpu_spmd_dataloader
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)
from transformers.optimization import Adafactor, get_scheduler
from transformers.processing_utils import ProcessorMixin
from transformers.pytorch_utils import (
    ALL_LAYERNORM_LAYERS,
    is_torch_greater_or_equal_than_2_3,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    ExportableState,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    DistributedTensorGatherer,
    EvalLoopContainer,
    IterableDatasetShard,
    LabelSmoother,
    LayerWiseDummyOptimizer,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_model_param_count,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
    remove_dummy_checkpoint,
    set_rng_state_for_device,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    HubStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    SaveStrategy,
    TrainerMemoryTracker,
    TrainOutput,
    check_target_module_exists,
    default_compute_objective,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    neftune_post_forward_hook,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.utils import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    XLA_FSDPV2_MIN_VERSION,
    PushInProgress,
    PushToHubMixin,
    can_return_loss,
    check_torch_load_is_safe,
    find_labels,
    is_accelerate_available,
    is_apex_available,
    is_apollo_torch_available,
    is_bitsandbytes_available,
    is_datasets_available,
    is_galore_torch_available,
    is_grokadamw_available,
    is_in_notebook,
    is_ipex_available,
    is_liger_kernel_available,
    is_lomo_available,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_schedulefree_available,
    is_torch_hpu_available,
    is_torch_mlu_available,
    is_torch_mps_available,
    is_torch_musa_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_xla_available,
    is_torch_xpu_available,
    is_torchao_available,
    logging,
    strtobool,
)
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.import_utils import requires
from transformers.utils.quantization_config import QuantizationMethod

# from transformers.training_args import OptimizerNames
# from transformers.utils import (
#     ADAPTER_CONFIG_NAME,
#     ADAPTER_SAFE_WEIGHTS_NAME,
#     ADAPTER_WEIGHTS_NAME,
#     CONFIG_NAME,
#     SAFE_WEIGHTS_INDEX_NAME,
#     SAFE_WEIGHTS_NAME,
#     WEIGHTS_INDEX_NAME,
#     WEIGHTS_NAME,
#     XLA_FSDPV2_MIN_VERSION,
#     PushInProgress,
#     PushToHubMixin,
#     can_return_loss,
#     check_torch_load_is_safe,
#     find_labels,
#     is_accelerate_available,
#     is_apollo_torch_available,
#     is_bitsandbytes_available,
#     is_datasets_available,
#     is_galore_torch_available,
#     is_grokadamw_available,
#     is_in_notebook,
#     is_ipex_available,
#     is_liger_kernel_available,
#     is_lomo_available,
#     is_peft_available,
#     is_safetensors_available,
#     is_sagemaker_dp_enabled,
#     is_sagemaker_mp_enabled,
#     is_schedulefree_available,
#     is_torch_hpu_available,
#     is_torch_mlu_available,
#     is_torch_mps_available,
#     is_torch_musa_available,
#     is_torch_neuroncore_available,
#     is_torch_npu_available,
#     is_torch_xla_available,
#     is_torch_xpu_available,
#     is_torchao_available,
#     logging,
#     strtobool,
# )
from tqdm import tqdm
if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.state import AcceleratorState
    from accelerate.utils import (
        AutocastKwargs,
        DistributedDataParallelKwargs,
        DistributedType,
        load_fsdp_model,
        load_fsdp_optimizer,
        save_fsdp_model,
        save_fsdp_optimizer,
    )
# from transformers.debug_utils import DebugOption
# from transformers.integrations.deepspeed import deepspeed_init
    
from .cl.lwf.lwf_pre import LWF
    

logger = get_logger()


class Trainer(SwiftMixin, HfTrainer):
    args: TrainingArguments

    @contextmanager
    def _patch_loss_function(self):
        model = self.model
        if isinstance(model, PeftModel):
            model = model.model
        model_cls = model.__class__
        if not hasattr(model_cls, 'loss_function'):
            yield
            return

        loss_function = model.loss_function
        _old_loss_function = model_cls.loss_function

        @staticmethod
        @wraps(loss_function)
        def new_loss_function(logits, labels, **kwargs):
            labels = labels.to(logits.device)  # fix device_map
            return loss_function(logits=logits, labels=labels, **kwargs)

        model_cls.loss_function = new_loss_function
        try:
            yield
        finally:
            model_cls.loss_function = _old_loss_function

    def train(self, *args, **kwargs):
        with self._patch_loss_function():
            return super().train(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        if inputs.get('labels') is not None:
            self._compute_acc(outputs, inputs['labels'])
        if num_items_in_batch is not None and self.model_accepts_loss_kwargs:
            loss /= self.args.gradient_accumulation_steps
        return (loss, outputs) if return_outputs else loss


class EmbeddingTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compute_metrics = self.calculate_metric
        self.preprocess_logits_for_metrics = None
        self.label_names = ['labels']

    def calculate_metric(self, eval_prediction: EvalPrediction) -> Dict[str, float]:
        from swift.plugin.loss import infonce_loss, calculate_paired_metrics, calculate_infonce_metrics
        if self.compute_loss_func is infonce_loss:
            return calculate_infonce_metrics(eval_prediction.predictions, eval_prediction.label_ids)
        else:
            return calculate_paired_metrics(eval_prediction.predictions, eval_prediction.label_ids)


class Seq2SeqTrainer(SwiftMixin, DataLoaderMixin, HfSeq2SeqTrainer):
    args: Seq2SeqTrainingArguments

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = True  # fix transformers>=4.46.2

        self.use_lwf = self.args.use_lwf
        self.alpha = 0.001
        # =============================================================================================================
        # 获取数据集和数据收集器,创建pre DataLoader
        if self.use_lwf:
            from torch.utils.data import DataLoader
            train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=self.data_collator,
                shuffle=False,  # 预计算时通常不需要打乱
            )

            
        if self.args.predict_with_generate:
            from swift.llm import PtEngine
            self.infer_engine = PtEngine.from_model_template(
                self.model, self.template, max_batch_size=self.args.per_device_eval_batch_size)
        self.jsonl_writer = JsonlWriter(os.path.join(self.args.output_dir, 'predict.jsonl'))
        

        if self.use_lwf:
            lwf_temperature = 2.0
            import copy
            previous_task_model = self.model
            # previous_task_model = copy.deepcopy(self.model)
            self.lwf = LWF(previous_task_model, temperature=lwf_temperature)
            logger.info("LWF has been successfully enabled.")
            print(f"======LWF 对象已在 trainer 中初始化。 LWF 已启用======: {self.lwf.enabled}")
            self.lwf.precompute_logits(train_dataloader, self.model.device, self.args.use_logits_to_keep)
                
        else:
            print("LWF 在此 trainer 实例中未使用。")
        

    @staticmethod
    def _predict_data_collator(batch):
        return {'_data': batch}

    @contextmanager
    def _patch_predict_with_generate(self):
        origin_mode = self.template.mode
        self.template.set_mode('pt')
        is_multimodal = self.model.model_meta.is_multimodal
        origin_data_collator = self.data_collator

        if is_multimodal:
            models = self.template.remove_post_encode_hook()
        self.data_collator = self._predict_data_collator
        try:
            yield
        finally:
            if is_multimodal:
                self.template.register_post_encode_hook(models)
            self.data_collator = origin_data_collator
            self.template.set_mode(origin_mode)

    def evaluate(self, *args, **kwargs):
        context = self._patch_predict_with_generate() if self.args.predict_with_generate else nullcontext()
        with context:
            res = super().evaluate(*args, **kwargs)
            gc_collect()
            return res

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys)
        from swift.llm import RequestConfig, InferRequest
        data_list = inputs['_data']
        labels_list = [InferRequest.remove_response(data['messages']) for data in data_list]
        resp_list = self.infer_engine.infer(
            data_list,
            RequestConfig(max_tokens=self.model.generation_config.max_new_tokens),
            use_tqdm=False,
            template=self.template)

        response_list = []
        jsonl_cache = []
        device = self.args.device
        for data, resp, labels in zip(data_list, resp_list, labels_list):
            response = resp.choices[0].message.content
            jsonl_cache.append({'response': response, 'labels': labels, **data})
            response_list.append(Serializer.to_tensor(resp.choices[0].message.content).to(device=device))
        self.jsonl_writer.append(jsonl_cache, gather_obj=True)
        labels_list = [Serializer.to_tensor(labels).to(device=device) for labels in labels_list]
        response_list = pad_sequence(response_list, batch_first=True, padding_value=0)
        labels_list = pad_sequence(labels_list, batch_first=True, padding_value=0)
        return None, response_list, labels_list
    
    def training_step(
        self, model: nn.Module, inputs: dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            if is_torch_xpu_available():
                torch.xpu.empty_cache()
            elif is_torch_mlu_available():
                torch.mlu.empty_cache()
            elif is_torch_musa_available():
                torch.musa.empty_cache()
            elif is_torch_npu_available():
                torch.npu.empty_cache()
            elif is_torch_mps_available():
                torch.mps.empty_cache()
            elif is_torch_hpu_available():
                logger.warning(
                    "`torch_empty_cache_steps` is set but HPU device/backend does not support empty_cache()."
                )
            else:
                torch.cuda.empty_cache()

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            from apex import amp

            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            # Finally we need to normalize the loss for reporting if GA loss bug is not fixed during compute loss
            if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
                loss = loss / self.args.gradient_accumulation_steps

            # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
            # https://github.com/huggingface/transformers/pull/35808
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False

            self.accelerator.backward(loss, **kwargs)
            
            # print(loss)
            # # 获取并打印梯度
            # for name, param in model.named_parameters():
            #     if param.requires_grad:  # 只处理可训练参数
            #         if param.grad is not None:
            #             print(f"可训练参数 {name} 的形状: {param.shape}")
            #             print(f"可训练参数 {name} 的梯度形状: {param.grad.shape}")
            #             # print(f"可训练参数 {name} 的梯度范数: {param.grad.norm()}")
            #         else:
            #             print(f"可训练参数 {name} 的梯度为 None（可能未计算梯度）")
            #     # else:
            #     #     print(f"参数 {name} 不可训练（requires_grad=False）")

            return loss.detach()


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if hasattr(self.args, 'EWC') and self.args.EWC:
            return self.compute_ewc_loss(model, inputs, return_outputs, num_items_in_batch)
        elif hasattr(self.args, 'GEM') and self.args.GEM:
            if hasattr(self.args, 'GEM_memory_strength') and self.args.GEM_memory_strength > 0:
                return self.compute_gem_loss(model, inputs, return_outputs, num_items_in_batch)
        from swift.plugin.loss import get_loss_func
        loss_kwargs = {}
        labels = None
        compute_loss_func = self.compute_loss_func
        loss_scale = inputs.pop('loss_scale', None)
        if loss_scale is not None:
            loss_kwargs['loss_scale'] = loss_scale
            if compute_loss_func is None:
                compute_loss_func = get_loss_func('loss_scale')
        if (self.label_smoother is not None or compute_loss_func is not None) and 'labels' in inputs:
            labels = inputs.pop('labels')

        base_model = self.template.get_base_model(self.model)
        use_logits_to_keep = self.args.use_logits_to_keep
        if use_logits_to_keep is None:
            # padding_free or packing
            use_logits_to_keep = 'labels' in inputs and 'logits_to_keep' in inspect.signature(
                base_model.forward).parameters
        logger.info_once(f'use_logits_to_keep: {use_logits_to_keep}')

        if use_logits_to_keep:
            if inputs['labels'].shape[0] == 1 and not is_mp():
                # device_map may encounter device mismatch issues.
                loss_mask = (inputs['labels'] != -100)[0]
                inputs['labels'] = inputs['labels'][:, loss_mask]
                inputs['labels'] = nn.functional.pad(inputs['labels'], (1, 0), value=-100)
                inputs['logits_to_keep'] = nn.functional.pad(loss_mask[1:], (0, 1), value=True)
            else:
                inputs['logits_to_keep'] = (inputs['labels'].shape[-1] -
                                            (torch.ne(inputs['labels'], -100).int().argmax(-1))).max().item() + 1
                assert inputs['logits_to_keep'] > 0
                inputs['labels'] = inputs['labels'][:, -inputs['logits_to_keep']:]
                
        if 'index' in inputs.keys():
            sample_ids = inputs.pop("index")
        else:
            sample_ids = None
            
        with self.template.compute_loss_context(self.model, inputs):
            outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is None:
            labels = inputs['labels']
            outputs.loss = outputs.loss.to(labels.device)
            # fix https://github.com/huggingface/transformers/issues/34263
            if num_items_in_batch is not None:
                outputs.loss = outputs.loss * (labels[:, 1:] != -100).sum() / num_items_in_batch

            if isinstance(outputs, dict) and 'loss' not in outputs:
                raise ValueError(
                    'The model did not return a loss from the inputs, only the following keys: '
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}.")
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
        else:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if is_peft_available() and isinstance(unwrapped_model, PeftModel):
                model_name = unwrapped_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if compute_loss_func is not None:
                loss = compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch, **loss_kwargs)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        
        if self.template.sequence_parallel_size > 1:
            from swift.trainers.sequence_parallel import sequence_parallel
            loss = sequence_parallel.reduce_outputs(loss, labels)

        if getattr(self.args, 'average_tokens_across_devices', False) and self.model_accepts_loss_kwargs:
            loss *= self.accelerator.num_processes
        if self.use_lwf:
            lwf_loss = self.ContinualLearningLWF(outputs, inputs, sample_ids)
            loss += lwf_loss

        if outputs.logits is not None and labels is not None:
            # Liger does not have logits
            self._compute_acc(outputs, labels)
        return (loss, outputs) if return_outputs else loss
    
    def ContinualLearningLWF(self, outputs, inputs, sample_ids):  
        # If LWF is enabled, add LWF loss
        # print(sample_ids)
        if self.use_lwf:
            if len(self.lwf.cached_logits) > 0:
                # print(f"[rank ] 使用内存中缓存的 logits 计算 LWF 损失。")
                lwf_loss = self.lwf.lwf_loss_with_cached_logits(outputs.logits, inputs, sample_ids)
            else:
                # print(f"[rank] 未找到缓存的 logits，使用实时计算 LWF 损失。")
                lwf_loss = self.lwf.lwf_loss(outputs.logits, inputs)

        return lwf_loss * self.alpha

    def compute_origin_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        from swift.plugin.loss import get_loss_func
        loss_kwargs = {}
        labels = None
        compute_loss_func = self.compute_loss_func
        loss_scale = inputs.pop('loss_scale', None)
        if loss_scale is not None:
            loss_kwargs['loss_scale'] = loss_scale
            if compute_loss_func is None:
                compute_loss_func = get_loss_func('loss_scale')
        if (self.label_smoother is not None or compute_loss_func is not None) and 'labels' in inputs:
            labels = inputs.pop('labels')
        base_model = self.template.get_base_model(self.model)
        use_logits_to_keep = self.args.use_logits_to_keep
        if use_logits_to_keep is None:
            # padding_free or packing
            use_logits_to_keep = 'labels' in inputs and 'logits_to_keep' in inspect.signature(
                base_model.forward).parameters
        logger.info_once(f'use_logits_to_keep: {use_logits_to_keep}')
        if use_logits_to_keep:
            if inputs['labels'].shape[0] == 1 and not is_mp():
                # device_map may encounter device mismatch issues.
                loss_mask = (inputs['labels'] != -100)[0]
                inputs['labels'] = inputs['labels'][:, loss_mask]
                inputs['labels'] = nn.functional.pad(inputs['labels'], (1, 0), value=-100)
                inputs['logits_to_keep'] = nn.functional.pad(loss_mask[1:], (0, 1), value=True)
            else:
                inputs['logits_to_keep'] = (inputs['labels'].shape[-1] -
                                            (torch.ne(inputs['labels'], -100).int().argmax(-1))).max().item() + 1
                assert inputs['logits_to_keep'] > 0
                inputs['labels'] = inputs['labels'][:, -inputs['logits_to_keep']:]
        with self.template.compute_loss_context(self.model, inputs):
            outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        if labels is None:
            labels = inputs['labels']
            outputs.loss = outputs.loss.to(labels.device)
            # fix https://github.com/huggingface/transformers/issues/34263
            if num_items_in_batch is not None:
                outputs.loss = outputs.loss * (labels[:, 1:] != -100).sum() / num_items_in_batch

            if isinstance(outputs, dict) and 'loss' not in outputs:
                raise ValueError(
                    'The model did not return a loss from the inputs, only the following keys: '
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}.")
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
        else:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if is_peft_available() and isinstance(unwrapped_model, PeftModel):
                model_name = unwrapped_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if compute_loss_func is not None:
                loss = compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch, **loss_kwargs)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        if self.template.sequence_parallel_size > 1:
            from swift.trainers.sequence_parallel import sequence_parallel
            loss = sequence_parallel.reduce_outputs(loss, labels)
        if getattr(self.args, 'average_tokens_across_devices', False) and self.model_accepts_loss_kwargs:
            loss *= self.accelerator.num_processes
        if outputs.logits is not None and labels is not None:
            # Liger does not have logits
            self._compute_acc(outputs, labels)
        return (loss, outputs) if return_outputs else loss


    def ewc_after_train(self):
        if self.args.local_rank not in [-1, 0]:
            return
        """支持多卡的EWC Fisher矩阵计算（含完整数据加载器初始化）"""
        original_dataset = self.get_train_dataloader().dataset
        data_len = min(len(original_dataset),self.args.EWC_limit)
        train_dataloader = DataLoader(
            dataset=original_dataset.select(range(data_len)),
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            shuffle=False
        )
        try:
            total_samples = len(train_dataloader.dataset)
        except (TypeError, AttributeError):
            if hasattr(train_dataloader.dataset, '__len__'):
                total_samples = len(train_dataloader.dataset)
        print(f"后训练样本数: {total_samples}")

        # 2. 初始化Fisher矩阵
        fisher = {
            n: torch.zeros_like(p, device='cpu') 
            for n, p in self.model.named_parameters() 
            if p.requires_grad
        }
        optpar = {
            n: p.detach().clone().cpu()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        self.model.train()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)

        with torch.enable_grad():
            # 添加进度条（动态描述+总步数+单位）
            progress_bar = tqdm(
                enumerate(train_dataloader),
                total=len(train_dataloader),
                desc="EWC计算",
                unit="batch"
            )
            
            for step, inputs in progress_bar:
                self.model.zero_grad()
                inputs = {k: v.to(device) for k, v in inputs.items()}  # 二次设备确认
                outputs = self.model(**inputs)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                loss = loss.mean()
                loss.backward()
                for n, p in self.model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        hp_grad = p.grad
                        fisher[n] += hp_grad.data.cpu() ** 2

                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    })



        # 关闭进度条（避免后续打印混乱）
        progress_bar.close()

        # 4. 结果保存
        if self.args.local_rank == 0 or self.args.local_rank == -1:
            fisher = {n: f / data_len for n, f in fisher.items()}  # 平均
            torch.save(fisher, os.path.join(self.args.output_dir, 'fisher.bin'))
            torch.save(optpar, os.path.join(self.args.output_dir, 'optpar.bin'))

    
    def compute_ewc_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        from swift.plugin.loss import get_loss_func
        loss_kwargs = {}
        labels = None
        compute_loss_func = self.compute_loss_func
        loss_scale = inputs.pop('loss_scale', None)
        if loss_scale is not None:
            loss_kwargs['loss_scale'] = loss_scale
            if compute_loss_func is None:
                compute_loss_func = get_loss_func('loss_scale')
        if (self.label_smoother is not None or compute_loss_func is not None) and 'labels' in inputs:
            labels = inputs.pop('labels')

        base_model = self.template.get_base_model(self.model)
        use_logits_to_keep = self.args.use_logits_to_keep
        if use_logits_to_keep is None:
            # padding_free or packing
            use_logits_to_keep = 'labels' in inputs and 'logits_to_keep' in inspect.signature(
                base_model.forward).parameters
        logger.info_once(f'use_logits_to_keep: {use_logits_to_keep}')

        if use_logits_to_keep:
            if inputs['labels'].shape[0] == 1 and not is_mp():
                # device_map may encounter device mismatch issues.
                loss_mask = (inputs['labels'] != -100)[0]
                inputs['labels'] = inputs['labels'][:, loss_mask]
                inputs['labels'] = nn.functional.pad(inputs['labels'], (1, 0), value=-100)
                inputs['logits_to_keep'] = nn.functional.pad(loss_mask[1:], (0, 1), value=True)
            else:
                inputs['logits_to_keep'] = (inputs['labels'].shape[-1] -
                                            (torch.ne(inputs['labels'], -100).int().argmax(-1))).max().item() + 1
                assert inputs['logits_to_keep'] > 0
                inputs['labels'] = inputs['labels'][:, -inputs['logits_to_keep']:]
        with self.template.compute_loss_context(self.model, inputs):
            outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is None:
            labels = inputs['labels']
            outputs.loss = outputs.loss.to(labels.device)
            # fix https://github.com/huggingface/transformers/issues/34263
            if num_items_in_batch is not None:
                outputs.loss = outputs.loss * (labels[:, 1:] != -100).sum() / num_items_in_batch

            if isinstance(outputs, dict) and 'loss' not in outputs:
                raise ValueError(
                    'The model did not return a loss from the inputs, only the following keys: '
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}.")
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
        else:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if is_peft_available() and isinstance(unwrapped_model, PeftModel):
                model_name = unwrapped_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if compute_loss_func is not None:
                loss = compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch, **loss_kwargs)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)

        if self.template.sequence_parallel_size > 1:
            from swift.trainers.sequence_parallel import sequence_parallel
            loss = sequence_parallel.reduce_outputs(loss, labels)
        dev = loss.device
        if hasattr(self.model, 'fisher') and hasattr(self.model, 'optpar'):
            ewc_loss = 0
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    # dev = p.device
                    # print(n, dev)
                    l = self.args.EWC_lambda * self.model.fisher[n].to(dev) * (p.data.to(dev) - self.model.optpar[n].to(dev)).pow(2)
                    ewc_loss += l.sum()
            loss += ewc_loss
            print('loss_ewc',ewc_loss)

        if getattr(self.args, 'average_tokens_across_devices', False) and self.model_accepts_loss_kwargs:
            loss *= self.accelerator.num_processes

        if outputs.logits is not None and labels is not None:
            # Liger does not have logits
            self._compute_acc(outputs, labels)
        return (loss, outputs) if return_outputs else loss
    

                
                
    def compute_gem_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Computes the loss for the current task, applying GEM projection if necessary.

        Args:
            model: The model to compute the loss for.
            inputs: The input batch. Must contain an 'is_memory' tensor.
            return_outputs: Whether to return model outputs alongside the loss.

        Returns:
            The computed loss (potentially after gradient modification), and optionally the model outputs.
        """
        rank = get_rank() # Get rank for logging
        is_memory = inputs["is_memory"]
        current_mask = ~is_memory
        memory_mask = is_memory

        # Handle cases where a batch might only contain current or memory samples
        has_current_samples = torch.any(current_mask)
        has_memory_samples = torch.any(memory_mask)

        if not has_current_samples:
            logger.warning_once(f"[RANK {rank}] Received a batch with only memory samples. Skipping GEM gradient calculation for this batch.")
            effective_loss = torch.tensor(1e-9, device=model.device, requires_grad=True) # MODIFIED DEFAULT to small epsilon
            if has_memory_samples:
                memory_inputs_for_dummy_pass = {
                    k: v[memory_mask]
                    for k, v in inputs.items()
                    if k != "is_memory" and isinstance(v, torch.Tensor) and v.shape[0] == inputs["is_memory"].shape[0]
                }
                memory_inputs_for_dummy_pass = {
                    k: val_masked for k, val_masked in memory_inputs_for_dummy_pass.items() if val_masked.numel() > 0
                }
                if memory_inputs_for_dummy_pass.get("input_ids") is not None: # 'input_ids' is crucial
                    loss_from_mem_samples = self.compute_origin_loss(model, memory_inputs_for_dummy_pass, return_outputs=False)
                    effective_loss_zero_contrib = loss_from_mem_samples * 0.0
                    effective_loss = effective_loss_zero_contrib + torch.tensor(1e-9, device=effective_loss_zero_contrib.device, requires_grad=False)
                    if loss_from_mem_samples.requires_grad and not effective_loss.requires_grad:
                        effective_loss = effective_loss.clone().detach().requires_grad_(True)
                else:
                    effective_loss = torch.tensor(1e-9, device=model.device, requires_grad=True)

            if self.args.max_grad_norm > 0 and hasattr(model, "parameters"):
                param_perturbed = False
                for param in model.parameters():
                    if param.requires_grad:
                        if effective_loss.requires_grad:
                            perturbation_coeff = 1e-12
                            effective_loss = effective_loss + (param.sum() * perturbation_coeff)
                            param_perturbed = True
                        else:
                            logger.warning_once(f"[RANK {get_rank()}] GEM compute_loss: effective_loss does not require_grad before perturbation attempt in memory-only batch. Skipping perturbation.")
                            break # Skip if base effective_loss is faulty
                
                if not param_perturbed and any(p.requires_grad for p in model.parameters()):
                    logger.warning_once(f"[RANK {get_rank()}] GEM compute_loss: Could not apply perturbation in memory-only batch (no suitable param or effective_loss issue), grad norm might still be zero.")

            if return_outputs:
                return (effective_loss, None)
            else:
                return effective_loss

        current_inputs = {k: v[current_mask] for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        current_inputs.pop('is_memory')
        loss_current_tuple = self.compute_origin_loss(model, current_inputs, return_outputs=True) # Request outputs
        loss_current = loss_current_tuple[0]
        outputs_current = loss_current_tuple[1]

        # Step 2: Compute gradient for the current task
        g_current = self._get_grads(model, loss_current)
        if g_current is None:
             return (loss_current.detach(), outputs_current) if return_outputs else loss_current.detach()

        # If no memory samples in this batch, behave like normal training
        if not has_memory_samples:
            return (loss_current, outputs_current) if return_outputs else loss_current

        # Step 3: Compute memory loss and gradient (only if memory samples exist)
        memory_inputs = {k: v[memory_mask] for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        memory_inputs.pop('is_memory')
        loss_memory = self.compute_origin_loss(model, memory_inputs, return_outputs=False)
        print('loss_memory',loss_memory)
        g_memory = self._get_grads(model, loss_memory)
        if g_memory is None:
            logger.warning_once(f"[RANK {rank}] Could not compute gradients for the memory samples in this batch. Skipping GEM projection.")
            return (loss_current, outputs_current) if return_outputs else loss_current
        # Step 4: Check for conflict and project if needed
        # Detach gradients before dot product to ensure it's just a value comparison
        g_current_detached = g_current.detach()
        g_memory_detached = g_memory.detach()
        dot_product = torch.dot(g_current_detached, g_memory_detached)
        if dot_product < 0:
            g_projected = self._project_gem_qp(g_current, [g_memory_detached]) # Pass detached memory grad to QP
            self._assign_grads(model, g_projected)
            loss_to_return = loss_current
        else:
            loss_to_return = loss_current

        return (loss_to_return, outputs_current) if return_outputs else loss_to_return

    def _get_grads(self, model: "PreTrainedModel", loss: torch.Tensor) -> Optional[torch.Tensor]:
        """Computes and returns the flattened gradients for the given loss."""
        rank = get_rank() # Get rank for logging
        if loss == 0.0 or not loss.requires_grad:
             return None # Cannot compute gradients if loss is zero or doesn't require grad

        # 检测DeepSpeed ZeRO阶段
        zero_stage = get_deepspeed_zero_stage(model)
        model.zero_grad() # Ensure grads are clean before backward
        loss.backward(retain_graph=True) # Retain graph is crucial for computing multiple grads
        # 对于ZeRO-2，需要同步梯度 (ZeRO-0 handled by DDP within backward)
        if zero_stage == 2:
            synchronize_gradients(model)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 收集梯度
        grads = []
        non_zero_grads_count = 0
        grads_l1_sum = 0.0

        # 对于ZeRO-3，需要使用gather_parameters上下文管理器
        gather_context = gather_parameters(model) if zero_stage == 3 else nullcontext()
        with gather_context:
            for name, param in model.named_parameters(): # Iterate with names for potential logging
                if param.requires_grad and param.grad is not None:
                    grad_view = param.grad.view(-1)
                    grads.append(grad_view.to(device=device))
                    grad_l1 = torch.sum(torch.abs(grad_view)).item()
                    grads_l1_sum += grad_l1
                    if grad_l1 > 1e-9: # Check if grad is effectively non-zero
                        non_zero_grads_count += 1
                elif param.requires_grad and param.grad is None:
                    pass 

        if not grads:
            logger.warning_once(f"[RANK {rank}] No gradients found for the model parameters.")
            return None
        # print(grads)
        flat_grads = torch.cat(grads).to(device)
        if zero_stage > 1: # Changed condition from is_distributed() and zero_stage > 0
            flat_grads = all_reduce_tensor(flat_grads)
        return flat_grads


    def _assign_grads(self, model: "PreTrainedModel", flat_grad: torch.Tensor):
        """Assigns the flattened gradient `flat_grad` back to the model parameters."""
        rank = get_rank() # Get rank for logging
        zero_stage = get_deepspeed_zero_stage(model)
        gather_context = gather_parameters(model) if zero_stage == 3 else nullcontext()
        with gather_context:
            pointer = 0
            assigned_count = 0
            for name, param in model.named_parameters(): # Iterate with names for potential logging
                if param.requires_grad and param.grad is not None: # Check grad is not None
                    numel = param.grad.numel()
                    if pointer + numel > flat_grad.numel():
                        break
                    # Ensure device compatibility before copying
                    param.grad.copy_(flat_grad[pointer : pointer + numel].view_as(param.grad).to(param.device))
                    pointer += numel
                    assigned_count += 1

        # 对于ZeRO-2，需要同步梯度 (ZeRO-0 handled by DDP)
        if zero_stage == 2:
            synchronize_gradients(model)

    def _project_gem_qp(self, g_current: torch.Tensor, memory_grads: List[torch.Tensor], max_iter: int = 15) -> torch.Tensor:
        """
        Projects the current gradient `g_current` to satisfy GEM constraints using LBFGS.
        Minimizes ||g' - g_current||^2 s.t. dot(g', g_mem_i) >= 0 for all i.

        Args:
            g_current: The current task's gradient (flattened).
            memory_grads: A list of gradients from memory tasks (flattened).
            max_iter: Max iterations for LBFGS.

        Returns:
            The projected gradient g'.
        """
        g_current = g_current.detach() # Ensure we don't modify the original grad tensor directly
        # Initialize the projected gradient as a parameter, starting from the current gradient
        g_proj = torch.nn.Parameter(g_current.clone(), requires_grad=True)
        # Use LBFGS optimizer on the projected gradient parameter
        optimizer = torch.optim.LBFGS([g_proj], max_iter=max_iter, line_search_fn="strong_wolfe")
        # Get the penalty strength from the instance variable
        penalty_strength = self.args.GEM_memory_strength
        # Define the closure for the optimizer
        def closure():
            # Closure rank might be useful if called across processes, though LBFGS usually runs locally
            optimizer.zero_grad()
            # Primary objective: minimize distance to original gradient
            loss = 0.5 * torch.norm(g_proj - g_current) ** 2
            # Constraint penalty: add penalty if dot product is negative
            for i, g_mem in enumerate(memory_grads):
                # Ensure memory gradient is detached and on the correct device
                g_mem_detached = g_mem.detach().to(g_proj.device)
                dot_prod = torch.dot(g_proj, g_mem_detached)
                # Only apply penalty if the constraint is violated (dot_prod < 0)
                if dot_prod < 0:
                    penalty = -penalty_strength * dot_prod
                    # Add penalty proportional to the violation magnitude
                    loss = loss + penalty # Maximize dot product towards >= 0 (original paper minimizes -dot, so equivalent to adding -strength * dot when dot < 0)
            loss.backward()
            return loss
        optimizer.step(closure)

        return g_proj.detach()
    
    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the initial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size
        (
            num_train_epochs,
            num_update_steps_per_epoch,
            num_examples,
            num_train_samples,
            epoch_based,
            len_dataloader,
            max_steps,
        ) = self.set_initial_training_values(args, train_dataloader, total_train_batch_size)

        num_train_tokens = None
        if self.args.include_tokens_per_second:
            num_train_tokens = self.num_tokens(train_dataloader, None if epoch_based else max_steps)
            # If going by epochs, multiply tokens linearly
            if len_dataloader is not None and epoch_based:
                num_train_tokens *= args.num_train_epochs
            # Otherwise since its steps, we just multiply by grad accum
            else:
                num_train_tokens *= args.gradient_accumulation_steps

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # Can't delay optimizer creation when using FSDP2: https://github.com/huggingface/accelerate/blob/3f636d626063ffcf9a337c7d3624d61b7d187d59/src/accelerate/accelerator.py#L1404
        is_fsdp2 = self.is_fsdp_enabled and (getattr(self.accelerator.state.fsdp_plugin, "fsdp_version", 1) == 2)
        if is_fsdp2:
            delay_optimizer_creation = False

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState(
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]
        )
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        self.state.compute_steps(args, max_steps)

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if use_accelerator_prepare and self.is_fsdp_enabled:
            # In case of auto_find_batch_size=True
            # Remove FSDP wrapping from sub-models.
            self.model = unwrap_model(self.model, recursive=True)

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                # configure fsdp plugin for qlora if any
                self._fsdp_qlora_plugin_updates()
                if self.accelerator.mixed_precision != "fp8":
                    self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )
        elif self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            # In this case we are in DDP + LOMO, which should be supported
            self.optimizer = self.accelerator.prepare(self.optimizer)

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
                )
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)
        self._load_scaler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            self._load_callback_state()
            epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        for attr in ("model", "optimizer", "lr_scheduler"):
            setattr(self.callback_handler, attr, getattr(self, attr))
        self.callback_handler.train_dataloader = train_dataloader

        self.state.init_training_references(self, max_steps, num_train_epochs, trial)

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0, device=args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None
        learning_rate = None
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        if args.eval_on_start:
            self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

        for epoch in range(epochs_trained, num_train_epochs):
            epoch_dataloader = train_dataloader
            if hasattr(epoch_dataloader, "set_epoch"):
                epoch_dataloader.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_dataloader)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_dataloader = skip_first_batches(epoch_dataloader, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            epoch_iterator = iter(epoch_dataloader)
            # We chunkify the epoch iterator into gradient accumulation steps `n` batches
            remainder = steps_in_epoch % args.gradient_accumulation_steps
            if remainder == 0:
                remainder = args.gradient_accumulation_steps
            update_step = -1
            total_updates = steps_in_epoch // args.gradient_accumulation_steps + int(
                remainder < args.gradient_accumulation_steps
            )
            for _ in range(total_updates):
                update_step += 1
                num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
                batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches, args.device)
                for i, inputs in enumerate(batch_samples):
                    step += 1
                    do_sync_step = (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == steps_in_epoch
                    # Since we perform prefetching, we need to manually set sync_gradients
                    self.accelerator.gradient_state._set_sync_gradients(do_sync_step)

                    if self.args.include_num_input_tokens_seen:
                        main_input_name = getattr(self.model, "main_input_name", "input_ids")
                        if main_input_name not in inputs:
                            logger.warning(
                                "Tried to track the number of tokens seen, however the current model is "
                                "not configured properly to know what item is the input. To fix this, add "
                                "a `main_input_name` attribute to the model class you are using."
                            )
                        else:
                            input_tokens = inputs[main_input_name].numel()
                            input_tokens = torch.tensor(input_tokens, device=self.args.device, dtype=torch.int64)
                            self.state.num_input_tokens_seen += self.accelerator.gather(input_tokens).sum().item()
                    if rng_to_sync:
                        self._load_rng_state(resume_from_checkpoint)
                        rng_to_sync = False

                    # Skip past any already trained steps if resuming training
                    if steps_trained_in_current_epoch > 0:
                        steps_trained_in_current_epoch -= 1
                        if steps_trained_progress_bar is not None:
                            steps_trained_progress_bar.update(1)
                        if steps_trained_in_current_epoch == 0:
                            self._load_rng_state(resume_from_checkpoint)
                        continue
                    elif steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.close()
                        steps_trained_progress_bar = None

                    if step % args.gradient_accumulation_steps == 0:
                        self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                    # We explicitly want to avoid relying on `accelerator.accumulate` for generation training
                    context = (
                        functools.partial(self.accelerator.no_sync, model=model)
                        if i != len(batch_samples) - 1
                        and self.accelerator.distributed_type != DistributedType.DEEPSPEED
                        else contextlib.nullcontext
                    )
                    with context():
                        tr_loss_step = self.training_step(model, inputs, num_items_in_batch)

                    if (
                        args.logging_nan_inf_filter
                        and not is_torch_xla_available()
                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                    ):
                        # if loss is nan or inf simply add the average of previous logged losses
                        tr_loss = tr_loss + tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                    else:
                        if tr_loss.device != tr_loss_step.device:
                            raise ValueError(
                                f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                            )
                        tr_loss = tr_loss + tr_loss_step

                    self.current_flos += float(self.floating_point_ops(inputs))

                    if do_sync_step:
                        # Since we perform prefetching, we need to manually set sync_gradients to True
                        self.accelerator.gradient_state._set_sync_gradients(True)

                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0:
                            if is_sagemaker_mp_enabled() and args.fp16:
                                _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
                            elif self.use_apex:
                                # Revert to normal clipping otherwise, handling Apex or full precision
                                _grad_norm = nn.utils.clip_grad_norm_(
                                    amp.master_params(self.optimizer),
                                    args.max_grad_norm,
                                )
                            else:
                                _grad_norm = self.accelerator.clip_grad_norm_(
                                    model.parameters(),
                                    args.max_grad_norm,
                                )

                            if (
                                is_accelerate_available()
                                and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                            ):
                                grad_norm = model.get_global_grad_norm()
                                # In some cases the grad norm may not return a float
                                if hasattr(grad_norm, "item"):
                                    grad_norm = grad_norm.item()
                            else:
                                grad_norm = _grad_norm

                        self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)

                        self.optimizer.step()
                        model.base_model.ema_update(model)

                        self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

                        # get leaning rate before update
                        learning_rate = self._get_learning_rate()

                        if not self.accelerator.optimizer_step_was_skipped:
                            # Delay optimizer scheduling until metrics are generated
                            if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                self.lr_scheduler.step()

                        model.zero_grad()
                        self.state.global_step += 1
                        self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                        self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                        self._maybe_log_save_evaluate(
                            tr_loss,
                            grad_norm,
                            model,
                            trial,
                            epoch,
                            ignore_keys_for_eval,
                            start_time,
                            learning_rate=learning_rate,
                        )
                    else:
                        self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                    # PyTorch/XLA relies on the data loader to insert the mark_step for
                    # each step. Since we are breaking the loop early, we need to manually
                    # insert the mark_step here.
                    if self.control.should_epoch_stop or self.control.should_training_stop:
                        if is_torch_xla_available():
                            xm.mark_step()
                        break
                # We also need to break out of the nested loop
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    if is_torch_xla_available():
                        xm.mark_step()
                    break
            if step < 0:
                logger.warning(
                    "There seems not to be a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(
                tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=learning_rate
            )

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_xla_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break
                
        model.base_model.ema_replace(model)

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_xla_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint, ignore_errors=True)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)