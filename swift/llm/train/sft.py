# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from functools import partial
import random
from typing import List, Union
import torch
from datasets import Dataset as HfDataset

from swift.plugin import extra_callbacks, get_loss_func, get_metric
from swift.trainers import TrainerFactory
from swift.utils import (append_to_jsonl, get_logger, get_model_parameter_info, is_master, plot_images, stat_array,
                         use_torchacc)
from ..argument import TrainArguments
from ..base import SwiftPipeline
from ..dataset import (EncodePreprocessor, GetLengthPreprocessor, IterablePackingDataset, LazyLLMDataset,
                       PackingDataset, load_dataset)
from ..infer import prepare_generation_config
from .tuner import TunerMixin
from swift.llm.train.distributed_utils import (
    is_distributed, is_main_process, get_rank, get_world_size,
    get_deepspeed_zero_stage, broadcast_object
)
logger = get_logger()


class SwiftSft(SwiftPipeline, TunerMixin):
    args_class = TrainArguments
    args: args_class

    def __init__(self, args: Union[List[str], TrainArguments, None] = None) -> None:
        super().__init__(args)
        self.train_msg = {}
        self._prepare_model_tokenizer()
        self._prepare_template()
        self._prepare_callbacks()

    def _prepare_generation_config(self):
        args = self.args
        self.model.origin_generation_config = self.model.generation_config
        self.model.generation_config = prepare_generation_config(self.model.generation_config,
                                                                 args.get_request_config(), self.tokenizer)
        logger.info(f'model.generation_config: {self.model.generation_config}')

    def _prepare_model_tokenizer(self):
        args = self.args
        if args.sequence_parallel_size > 1:
            from swift.trainers.sequence_parallel import sequence_parallel
            sequence_parallel.init_sequence_parallel(args.sequence_parallel_size)
        self.model, self.processor = args.get_model_processor()

        if hasattr(self.model, 'hf_device_map'):
            logger.info(f'model.hf_device_map: {self.model.hf_device_map}')

        logger.info(f'model_info: {self.model.model_info}')

        self._prepare_generation_config()

    def _prepare_template(self) -> None:
        template = self.args.get_template(self.processor)
        if self.args.task_type == 'causal_lm':
            template.set_mode('train')
        if template.use_model:
            template.model = self.model
        self.template = template

    def _get_gem_dataset(self):
        """
        获取数据集，支持GEM持续学习
        1. 加载当前任务的训练和验证数据集
        2. 如果启用GEM，加载历史任务数据作为记忆
        3. 合并数据集并更新相关参数
        """
        args = self.args
        dataset_kwargs = args.get_dataset_kwargs()
        
        # 加载当前任务的数据集
        train_dataset, val_dataset = load_dataset(
            args.dataset, 
            split_dataset_ratio=args.split_dataset_ratio, 
            shuffle=args.dataset_shuffle, 
            **dataset_kwargs
        )

        # 处理验证集
        if len(args.val_dataset) > 0:
            _, val_dataset = load_dataset(
                args.val_dataset, 
                split_dataset_ratio=1.0, 
                shuffle=args.val_dataset_shuffle, 
                **dataset_kwargs
            )
            assert args.split_dataset_ratio == 0.
        
        # GEM相关处理
        if hasattr(args, 'GEM') and args.GEM:
            logger.info("Using GEM (Gradient Episodic Memory) for continual learning")
            # 1. 加载历史任务数据（记忆数据）
            merged_datasets = []
            if hasattr(args, 'GEM_replay_task_list') and args.GEM_replay_task_list:
                memory_task_list = [os.path.join(self.args.GEM_previous_task_dataset, task.strip()) for task in args.GEM_replay_task_list.split(',')]
                GEM_maxsamples_list = None
                if args.GEM_maxsamples_list:
                    try:
                        GEM_maxsamples_list = [int(x.strip()) for x in args.GEM_maxsamples_list.split(',')]
                        if len(GEM_maxsamples_list) != len(memory_task_list):
                            logger.warning(f"Length of GEM_maxsamples_list ({len(GEM_maxsamples_list)}) doesn't match memory_task_list ({len(memory_task_list)}). Will use GEM_replay_ratio instead.")
                            GEM_maxsamples_list = None
                    except ValueError:
                        logger.warning(f"Invalid format in GEM_maxsamples_list: {args.GEM_maxsamples_list}. Will use GEM_replay_ratio instead.")
                        GEM_maxsamples_list = None

                for task_idx, memory_task in enumerate(memory_task_list):
                    # 加载记忆任务数据集
                    memory_dataset, _ = load_dataset(
                        memory_task,
                        split_dataset_ratio=0.0,  # 不需要分割
                        shuffle=False,  # 保持原始顺序
                        **dataset_kwargs
                    )
                    print(len(memory_dataset))
                    total_samples = len(memory_dataset)
                    
                    # Determine max samples based on GEM_maxsamples_list or GEM_replay_ratio
                    if GEM_maxsamples_list and task_idx < len(GEM_maxsamples_list):
                        max_samples = min(GEM_maxsamples_list[task_idx], total_samples)
                        logger.info(f"Using max samples from list: {max_samples}")
                    else:
                        max_samples = int(total_samples * args.GEM_replay_ratio)
                        logger.info(f"Using ratio-based max samples: {max_samples}")

                    if max_samples < total_samples:
                        # 在分布式环境中，确保所有进程选择相同的样本
                        if is_distributed():
                            # 设置相同的随机种子以确保一致性
                            random_seed = args.seed + task_idx
                            random.seed(random_seed)
                            torch.manual_seed(random_seed)
                            # 主进程选择样本并广播给其他进程
                            if is_main_process():
                                indices = random.sample(range(total_samples), max_samples)
                            else:
                                indices = None
                            # 广播索引到所有进程
                            indices = broadcast_object(indices)
                        else:
                            # 非分布式环境，直接选择样本
                            indices = random.sample(range(total_samples), max_samples)
                        memory_dataset_raw = memory_dataset.select(indices)
                    else:
                        memory_dataset_raw = memory_dataset
                    # Add 'is_memory': True flag to memory data
                    memory_dataset = memory_dataset_raw.map(lambda example: {"is_memory": True})
                    merged_datasets.append(memory_dataset)

                        
            train_dataset = train_dataset.map(lambda example: {"is_memory": False})
            
            if len(merged_datasets) > 0: # Should always be at least 1 if current task data exists
                original_size = len(train_dataset)
                total_memory_size = sum(len(d) for d in merged_datasets)
                # 将当前任务数据集添加到合并列表的开头
                merged_datasets.insert(0, train_dataset)
                # 检查数据集类型并选择合适的合并方法
                first_dataset = merged_datasets[0]
                if hasattr(first_dataset, '__class__') and first_dataset.__class__.__name__ == 'Dataset':
                    # HuggingFace datasets
                    try:
                        from datasets import concatenate_datasets
                        train_dataset = concatenate_datasets(merged_datasets)
                    except ImportError:
                        # 如果没有 concatenate_datasets，使用逐个合并
                        train_dataset = merged_datasets[0]
                        for dataset in merged_datasets[1:]:
                            if hasattr(train_dataset, 'concatenate'):
                                train_dataset = train_dataset.concatenate(dataset)
                            else:
                                # 转换为列表再合并
                                train_list = list(train_dataset)
                                train_list.extend(list(dataset))
                                train_dataset = train_list
                
                elif isinstance(first_dataset, list):
                    # 列表格式的数据集
                    train_dataset = []
                    for dataset in merged_datasets:
                        if isinstance(dataset, list):
                            train_dataset.extend(dataset)
                        else:
                            # 尝试转换为列表
                            train_dataset.extend(list(dataset))
                
                elif hasattr(first_dataset, '__iter__'):
                    # 可迭代对象
                    from itertools import chain
                    train_dataset = list(chain(*merged_datasets))

                if hasattr(train_dataset, 'shuffle'):
                    train_dataset = train_dataset.shuffle(seed=args.seed)
                elif isinstance(train_dataset, list):
                    random.seed(args.seed)
                    random.shuffle(train_dataset)

        print(f"样本批次: {len(train_dataset)}")
        return train_dataset, val_dataset

    def _get_dataset(self):
        # The random shuffling of the training set occurs in the dataloader of the trainer.
        args = self.args
        print('='*100)
        print(args.dataset)
        if args.replay_datasets:
            print(args.replay_datasets)
        print('='*100)
        dataset_kwargs = args.get_dataset_kwargs()
        train_dataset, val_dataset = load_dataset(
            args.dataset, split_dataset_ratio=args.split_dataset_ratio, shuffle=args.dataset_shuffle, **dataset_kwargs)
                

        if len(args.val_dataset) > 0:
            # Loading val dataset
            _, val_dataset = load_dataset(
                args.val_dataset, split_dataset_ratio=1.0, shuffle=args.val_dataset_shuffle, **dataset_kwargs)
            assert args.split_dataset_ratio == 0.
        logger.info(f'train_dataset: {train_dataset}')
        logger.info(f'val_dataset: {val_dataset}')
        
        # 初始化合并后的重放数据集列表
        replay_datasets_list = []
        # print(len(args.replay_datasets), '='*50)

        # if len(args.replay_datasets) > 0:
        #     replay_ratio = 0.5
        #     for replay_dataset in args.replay_datasets:
        #         # 加载并分割重放数据集，只保留训练部分
        #         replay_train, _ = load_dataset(replay_dataset, split_dataset_ratio=replay_ratio, shuffle=args.dataset_shuffle, **dataset_kwargs)
        #         replay_datasets_list.append(replay_train)  # 将每个重放数据集的训练部分添加到列表

        #     # 合并所有重放数据集
        #     from datasets import concatenate_datasets
        #     replay_datasets_list.append(train_dataset)
        #     if replay_datasets_list:
        #         train_replay_datasets = concatenate_datasets(replay_datasets_list)
        #         logger.info(f'train_replay_datasets: {train_replay_datasets}')
        #     else:
        #         train_replay_datasets = None  # 或使用原始 train_dataset
        #     print(len(train_replay_datasets), '='*50)
        #     return train_replay_datasets, val_dataset
        
        if len(args.replay_datasets) > 0:
            replay_ratio = 0
            for replay_dataset in args.replay_datasets:
                # num_samples = int(len(train_dataset) * 0.2)
                num_samples = 20
                # 加载并分割重放数据集，只保留训练部分
                replay_train, _ = load_dataset(replay_dataset, split_dataset_ratio=replay_ratio, shuffle=args.dataset_shuffle, **dataset_kwargs)
                indices = random.sample(range(len(replay_train)), num_samples)
                # 根据索引选择样本
                sampled_dataset = replay_train.select(indices)
                replay_datasets_list.append(sampled_dataset)  # 将每个重放数据集的训练部分添加到列表

            # 合并所有重放数据集
            from datasets import concatenate_datasets
            replay_datasets_list.append(train_dataset)
            if replay_datasets_list:
                train_replay_datasets = concatenate_datasets(replay_datasets_list)
                logger.info(f'train_replay_datasets: {train_replay_datasets}')
            else:
                train_replay_datasets = None  # 或使用原始 train_dataset
            print(len(train_replay_datasets), '='*50)
            return train_replay_datasets, val_dataset

        return train_dataset, val_dataset

    def _get_data_collator(self):
        args = self.args
        template = self.template
        padding_to = args.max_length if args.train_type == 'longlora' else None
        return partial(template.data_collator, padding_to=padding_to)

    @staticmethod
    def _save_val_dataset(output_dir: str, val_dataset):
        if is_master() and isinstance(val_dataset, HfDataset):
            os.makedirs(output_dir, exist_ok=True)
            val_dataset_path = os.path.join(output_dir, 'val_dataset.jsonl')
            append_to_jsonl(val_dataset_path, val_dataset.to_list())
            logger.info(f'The split dataset from the training set will be saved at: {val_dataset_path}.')

    def run(self):
        args = self.args
        if args.EWC:
            self.args.training_args.EWC = args.EWC
            self.args.training_args.EWC_lambda = args.EWC_lambda
            self.args.training_args.EWC_limit = args.EWC_limit
        if args.GEM:
            self.args.training_args.GEM = args.GEM
            self.args.training_args.GEM_memory_strength = args.GEM_memory_strength
            self.args.training_args.GEM_replay_ratio = args.GEM_replay_ratio
            self.args.training_args.GEM_replay_task_list = args.GEM_replay_task_list
            self.args.training_args.GEM_maxsamples_list = args.GEM_maxsamples_list
            self.args.training_args.GEM_previous_task_dataset = args.GEM_previous_task_dataset
        if args.GEM:
            train_dataset, val_dataset = self._get_gem_dataset()
            train_is_memory = train_dataset["is_memory"]
            train_dataset, val_dataset = self._encode_dataset(train_dataset, val_dataset)
            train_dataset = train_dataset.add_column("is_memory", train_is_memory)
        else:
            train_dataset, val_dataset = self._get_dataset()
            train_dataset, val_dataset = self._encode_dataset(train_dataset, val_dataset)
        
        # 使用 Hugging Face Dataset 的 map 方法添加索引
        import torch
        def add_index(example, idx):
            # 在每个样本中添加 'index' 字段
            example['index'] =  torch.tensor(idx)
            return example

        # 应用映射函数，with_indices=True 会传入样本索引
        train_dataset = train_dataset.map(add_index, with_indices=True)

        # # 验证 ID 是否已添加
        # if len(train_dataset) > 0 and 'index' in train_dataset.features:
        #     print(f"示例数据 (添加 ID 后1): {train_dataset[0].keys()}")


        if args.task_type == 'seq_cls':
            args.problem_type = args.problem_type or getattr(self.model.config, 'problem_type', None)
            logger.info(f'args.problem_type: {args.problem_type}')
        args.save_args()

        data_collator = self._get_data_collator()
        # Some tuners require train_dataset and data_collator for preparation: LoRA-GA
        self.model = self.prepare_model(self.args, self.model, template=self.template, train_dataset=train_dataset)
        logger.info(f'model: {self.model}')
        model_parameter_info = get_model_parameter_info(self.model)
        self.train_msg['model_parameter_info'] = model_parameter_info
        logger.info(f'model_parameter_info: {model_parameter_info}')
        
        trainer_cls = TrainerFactory.get_trainer_cls(args)
        trainer = trainer_cls(
            model=self.model,
            args=self.args.training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=self.callbacks,
            template=self.template,
            **self._get_trainer_kwargs(),
        )
        return self.train(trainer)

    def _get_trainer_kwargs(self):
        args = self.args
        if args.metric is not None:
            compute_metrics, preprocess_logits_for_metrics = get_metric(args.metric)
        elif args.predict_with_generate:
            compute_metrics, preprocess_logits_for_metrics = get_metric('nlg')
        else:
            compute_metrics, preprocess_logits_for_metrics = get_metric('acc')
            compute_metrics = partial(
                compute_metrics, acc_strategy=args.acc_strategy, is_encoder_decoder=self.template.is_encoder_decoder)
        return {
            'compute_metrics': compute_metrics,
            'preprocess_logits_for_metrics': preprocess_logits_for_metrics,
            'compute_loss_func': get_loss_func(args.loss_type)
        }

    def _save_trainer_state(self, trainer):
        training_args = trainer.args
        state = trainer.state
        if hasattr(state, 'last_model_checkpoint'):
            if is_master() and self.args.create_checkpoint_symlink:
                last_checkpoint = os.path.join(self.args.output_dir, 'last')
                best_checkpoint = os.path.join(self.args.output_dir, 'best')
                os.symlink(state.last_model_checkpoint, last_checkpoint)
                os.symlink(state.best_model_checkpoint, best_checkpoint)
                state.last_model_checkpoint = last_checkpoint
                state.best_model_checkpoint = best_checkpoint
        else:
            state.last_model_checkpoint = None
        logger.info(f'last_model_checkpoint: {state.last_model_checkpoint}')
        logger.info(f'best_model_checkpoint: {state.best_model_checkpoint}')

        # Visualization
        if is_master() and not use_torchacc():
            if 'tensorboard' in training_args.report_to:
                images_dir = os.path.join(training_args.output_dir, 'images')
                logger.info(f'images_dir: {images_dir}')
                plot_images(images_dir, training_args.logging_dir, ['train/loss'], 0.9)
            if training_args.push_to_hub:
                trainer.push_to_hub()

        self.train_msg.update({
            'last_model_checkpoint': state.last_model_checkpoint,
            'best_model_checkpoint': state.best_model_checkpoint,
            'best_metric': state.best_metric,
            'global_step': state.global_step,
            'log_history': state.log_history,
            'memory': trainer.max_memory,
        })
        if is_master():
            jsonl_path = os.path.join(training_args.output_dir, 'logging.jsonl')
            append_to_jsonl(jsonl_path, self.train_msg)
        return self.train_msg

    def train(self, trainer):
        logging_path = os.path.join(trainer.args.output_dir, 'logging.jsonl')
        logger.info(f'The logging file will be saved in: {logging_path}')
        if self.args.EWC and self.args.EWC_path is not None:
            fisher = torch.load(os.path.join(self.args.EWC_path, 'fisher.bin'), map_location='cpu')
            optpar = torch.load(os.path.join(self.args.EWC_path, 'optpar.bin'), map_location='cpu')
            fisher = {(k[6:] if k.startswith('model') else k): v for k, v in fisher.items()}
            optpar = {(k[6:] if k.startswith('model') else k): v for k, v in optpar.items()}
            self.model.fisher = fisher
            self.model.optpar = optpar
        try:
            trainer.train(trainer.args.resume_from_checkpoint)
        finally:
            res = self._save_trainer_state(trainer)
        if self.args.EWC:
            trainer.ewc_after_train()
        return res

    def _prepare_callbacks(self):
        from .callback import DynamicLayerActivationCallback, TrainerAdapterCallback
        args = self.args
        callbacks = []
        if args.lisa_activated_layers > 0:
            assert args.train_type == 'full', 'LISA only supports full parameter training.'
            lisa_callback = DynamicLayerActivationCallback(
                n_layers=args.lisa_activated_layers,  # Number of layers to activate
                step_interval=args.lisa_step_interval,  # Step interval to update active layers
                model=self.model)
            lisa_callback.switch_active_layers()  # Make trainable parameters printing a correct value
            callbacks.append(lisa_callback)

        if args.is_adapter and args.train_type == 'adalora':
            callbacks.append(TrainerAdapterCallback(args))
        callbacks += extra_callbacks
        self.callbacks = callbacks

    def _stat_dataset(self, dataset: Union[HfDataset, PackingDataset]):
        args = self.args
        if isinstance(dataset, HfDataset):
            dataset = GetLengthPreprocessor()(dataset, num_proc=args.dataset_num_proc)
            length = dataset['length']
        else:
            length = dataset.packed_dataset.length_list
        _, stat_str = stat_array(length)
        logger.info(f'Dataset Token Length: {stat_str}')
        return stat_str

    def _encode_dataset(self, train_dataset, val_dataset):
        template = self.template
        args = self.args
        output_dir = getattr(args, 'output_dir', None) or getattr(args, 'save')
        self._save_val_dataset(output_dir, val_dataset)
        is_grpo = hasattr(args, 'rlhf_type') and args.rlhf_type == 'grpo'
        predict_with_generate = getattr(args, 'predict_with_generate', False)
        if not is_grpo:
            if args.packing:
                packing_dataset_cls = IterablePackingDataset if args.streaming else PackingDataset
                train_dataset = packing_dataset_cls(
                    self.template, train_dataset, num_proc=args.dataset_num_proc, strict=args.strict)
                if val_dataset is not None:
                    val_dataset = packing_dataset_cls(
                        self.template, val_dataset, num_proc=args.dataset_num_proc, strict=args.strict)
            elif args.lazy_tokenize:
                train_dataset = LazyLLMDataset(
                    train_dataset, template.encode, strict=args.strict, random_state=args.data_seed)
                if val_dataset is not None and not predict_with_generate:
                    val_dataset = LazyLLMDataset(
                        val_dataset, template.encode, strict=args.strict, random_state=args.data_seed)
            else:
                preprocessor = EncodePreprocessor(template=template)
                train_dataset = preprocessor(
                    train_dataset,
                    num_proc=args.dataset_num_proc,
                    load_from_cache_file=args.load_from_cache_file,
                    strict=args.strict)
                if val_dataset is not None and not predict_with_generate:
                    val_dataset = preprocessor(
                        val_dataset,
                        num_proc=args.dataset_num_proc,
                        load_from_cache_file=args.load_from_cache_file,
                        strict=args.strict)

            if is_master():
                inputs = train_dataset[0] if hasattr(train_dataset, '__len__') else next(iter(train_dataset))
                template.print_inputs(inputs, tokenizer_kwargs=inputs.pop('tokenizer_kwargs', None) or {})
            elif hasattr(train_dataset, '__len__'):
                # Avoid the random mismatch issue in LazyLLMDataset.
                inputs = train_dataset[0]
            if isinstance(train_dataset, (HfDataset, PackingDataset)):
                self.train_msg['train_dataset'] = self._stat_dataset(train_dataset)
                if val_dataset is not None and not predict_with_generate:
                    self.train_msg['val_dataset'] = self._stat_dataset(val_dataset)

        return train_dataset, val_dataset


def sft_main(args: Union[List[str], TrainArguments, None] = None):
    return SwiftSft(args).main()
