import os
import io
from contextlib import nullcontext
from typing import Any, List, Optional

import torch
def debugprint(*args, **kwargs):
    pass


# ======== 基础分布式检测 ======== #
def is_distributed() -> bool:
    """是否已初始化分布式环境"""
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_rank() -> int:
    """全局 rank（未初始化时先读环境变量，默认 0）"""
    if is_distributed():
        return torch.distributed.get_rank()
    return int(os.getenv("RANK", 0))


def get_local_rank() -> int:
    """节点内 local-rank（未初始化时先读环境变量，默认 0）"""
    if is_distributed():
        return torch.distributed.get_rank(group=torch.distributed.group.WORLD)
    return int(os.getenv("LOCAL_RANK", 0))


def is_main_process() -> bool:
    """是否为主进程（rank==0）"""
    return get_rank() == 0


def get_world_size() -> int:
    """进程总数（未初始化时先读环境变量，默认 1）"""
    if is_distributed():
        return torch.distributed.get_world_size()
    return int(os.getenv("WORLD_SIZE", 1))


# ======== DeepSpeed 相关 ======== #
def is_deepspeed_zero3_enabled() -> bool:
    """Transformers 环境下的 ZeRO-3 检测（仅供参考）"""
    try:
        from transformers.integrations import is_deepspeed_zero3_enabled

        return is_deepspeed_zero3_enabled()
    except (ImportError, RuntimeError):
        return False


def get_deepspeed_zero_stage(model) -> int:
    """
    通用获取 ZeRO-Stage：
      0  → 未启用 DeepSpeed
      1/2/3 → 对应 stage
    """
    rank = get_rank()
    debugprint(f"[rank {rank}] 检测 DeepSpeed ZeRO-Stage")

    # ① model 本身是 DeepSpeedEngine
    if hasattr(model, "zero_optimization_stage"):
        stage = model.zero_optimization_stage()
        debugprint(f"[rank {rank}] model.zero_optimization_stage() = {stage}")
        return stage

    # ② 外层封装（PEFT/LoRA/代理模块）
    if hasattr(model, "ds_engine") and hasattr(model.ds_engine, "zero_optimization_stage"):
        stage = model.ds_engine.zero_optimization_stage()
        debugprint(f"[rank {rank}] model.ds_engine.zero_optimization_stage() = {stage}")
        return stage

    # ③ Transformers 全局 DeepSpeed 配置
    try:
        from transformers.integrations import deepspeed_config

        cfg = deepspeed_config()
        if cfg and "zero_optimization" in cfg:
            stage = int(cfg["zero_optimization"].get("stage", 0))
            debugprint(f"[rank {rank}] transformers.deepspeed_config stage = {stage}")
            return stage
    except (ImportError, RuntimeError):
        pass

    # ④ 环境变量兜底
    if "DEEPSPEED_ZERO_STAGE" in os.environ:
        stage = int(os.environ["DEEPSPEED_ZERO_STAGE"])
        debugprint(f"[rank {rank}] 环境变量 DEEPSPEED_ZERO_STAGE = {stage}")
        return stage

    # ⑤ 参数特征（仅识别 stage-3）
    try:
        for name, p in model.named_parameters():
            if hasattr(p, "ds_id"):
                debugprint(f"[rank {rank}] 参数 {name} 含 ds_id → Stage-3")
                return 3
            break
    except Exception as e:
        debugprint(f"[rank {rank}] 检查参数 ds_id 失败: {e}")

    debugprint(f"[rank {rank}] 未检测到 DeepSpeed → Stage-0")
    return 0


# ======== ZeRO 专用工具 ======== #
def synchronize_gradients(model) -> None:
    """仅在 ZeRO-2 下同步分片梯度"""
    if hasattr(model, "ds_engine") and get_deepspeed_zero_stage(model) == 2:
        debugprint("[ZeRO-2] synchronize_gradients()")
        model.ds_engine.optimizer.synchronize_gradients()

def gather_parameters(model, params: Optional[List[torch.nn.Parameter]] = None):
    """
    在 ZeRO-3 下临时 All-Gather 完整参数的上下文管理器。
    其他场景返回 nullcontext。
    """
    from contextlib import nullcontext
    if get_deepspeed_zero_stage(model) == 3:
        try:
            import deepspeed

            if params is None:
                params = [p for p in model.parameters() if p.requires_grad]

            debugprint(f"[rank {get_rank()}] Gather {len(params)} params (ZeRO-3)")
            return deepspeed.zero.GatheredParameters(params, modifier_rank=None)
        except Exception as e:
            debugprint(f"[rank {get_rank()}] GatheredParameters 出错: {e}")
            return nullcontext()
    else:
        return nullcontext()



# ======== 通用通信工具 ======== #
def all_reduce_tensor(
    tensor: torch.Tensor, op: torch.distributed.ReduceOp = torch.distributed.ReduceOp.SUM
) -> torch.Tensor:
    """
    对张量执行 all_reduce；若 op==SUM 则自动取平均。
    非分布式环境直接返回。
    """
    if not is_distributed():
        return tensor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not tensor.is_cuda:
        tensor = tensor.to(device)

    orig_dtype = tensor.dtype
    if tensor.dtype != torch.float32:
        tensor = tensor.float()

    torch.distributed.all_reduce(tensor, op=op)

    if op == torch.distributed.ReduceOp.SUM:
        tensor /= get_world_size()

    if orig_dtype != torch.float32:
        tensor = tensor.to(orig_dtype)

    return tensor


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """
    广播任意 Python 对象；非分布式环境直接返回。
    """
    if not is_distributed():
        return obj

    rank = get_rank()
    world_size = get_world_size()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    debugprint(f"[DEBUG] broadcast_object: [rank {rank}/{world_size-1}] 开始广播对象，源进程: {src}")
    debugprint(f"[rank {rank}/{world_size-1}] broadcast_object 开始，源进程: {src}")

    # 序列化对象
    if rank == src:
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        data = buffer.getvalue()
        data_size = len(data)
        debugprint(f"[DEBUG] broadcast_object: [rank {rank}/{world_size-1}] 对象序列化完成，大小: {data_size/1024/1024:.2f} MB")
        debugprint(f"[rank {rank}/{world_size-1}] 对象序列化完成，大小: {data_size/1024/1024:.2f} MB")
    else:
        data = None
        data_size = 0
        debugprint(f"[DEBUG] broadcast_object: [rank {rank}/{world_size-1}] 准备接收对象")

    # 广播长度
    size = torch.tensor([0], dtype=torch.long, device=device)
    if rank == src:
        size[0] = torch.tensor([len(data)], dtype=torch.long, device=device)
        debugprint(f"[DEBUG] broadcast_object: [rank {rank}/{world_size-1}] 准备广播数据大小: {size.item()} 字节")

    # 添加每个进程的状态打印
    for i in range(world_size):
        if rank == i:
            debugprint(f"[DEBUG] broadcast_object: [rank {rank}/{world_size-1}] 准备广播/接收数据大小")
        

    torch.distributed.broadcast(size, src)

    debugprint(f"[DEBUG] broadcast_object: [rank {rank}/{world_size-1}] 已接收数据大小: {size.item()} 字节")
    debugprint(f"[rank {rank}/{world_size-1}] 已接收数据大小: {size.item()} 字节")

    # 广播数据
    if rank == src:
        tensor = torch.ByteTensor(list(data)).to(device)
        debugprint(f"[DEBUG] broadcast_object: [rank {rank}/{world_size-1}] 准备广播数据内容...")
    else:
        tensor = torch.empty(size.item(), dtype=torch.uint8, device=device)
        debugprint(f"[DEBUG] broadcast_object: [rank {rank}/{world_size-1}] 准备接收 {size.item()} 字节数据...")

    # 添加每个进程的状态打印
    for i in range(world_size):
        if rank == i:
            debugprint(f"[DEBUG] broadcast_object: [rank {rank}/{world_size-1}] 准备广播/接收数据内容")

    torch.distributed.broadcast(tensor, src)

    debugprint(f"[DEBUG] broadcast_object: [rank {rank}/{world_size-1}] 数据广播/接收完成")
    debugprint(f"[rank {rank}/{world_size-1}] 数据广播/接收完成")

    # 反序列化
    if rank != src:
        try:
            debugprint(f"[DEBUG] broadcast_object: [rank {rank}/{world_size-1}] 开始反序列化数据...")
            buffer = io.BytesIO(tensor.cpu().numpy().tobytes())
            obj = torch.load(buffer)
            debugprint(f"[DEBUG] broadcast_object: [rank {rank}/{world_size-1}] 数据反序列化成功")
            debugprint(f"[rank {rank}/{world_size-1}] 数据反序列化成功")
        except Exception as e:
            debugprint(f"[DEBUG] broadcast_object: [rank {rank}/{world_size-1}] 数据反序列化失败: {str(e)}")
            debugprint(f"[rank {rank}/{world_size-1}] 数据反序列化失败: {str(e)}")
            raise

    debugprint(f"[DEBUG] broadcast_object: [rank {rank}/{world_size-1}] 广播对象完成")
    debugprint(f"[rank {rank}/{world_size-1}] broadcast_object 完成")
    return obj
