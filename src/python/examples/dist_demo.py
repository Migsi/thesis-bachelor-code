import os

import torch
import torch.distributed as dist

def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl", device_id=local_rank)
    else:
        device = torch.device("cpu")
        dist.init_process_group(backend="gloo")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    return rank, world_size, local_rank, device

def info_and_test(rank, local_rank, world_size, device):
    device_name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else "cpu"

    print(f"[Rank local/global/world_size]")
    print(f"[Rank {local_rank}/{rank}/{world_size}] Using device: {device_name}")

    # Test simple tensor
    x = torch.randn(2, 2, device=device)
    y = torch.randn(2, 2, device=device)
    z = x + y
    print(f"[Rank {local_rank}/{rank}/{world_size}] Sample tensor operation result:\n\t{z}")

    # Test sync
    dist.barrier()

    # Test all_reduce
    a = torch.tensor([rank], device=device)
    dist.all_reduce(a)
    print(f"[Rank {local_rank}/{rank}/{world_size}] Sample tensor all_reduce result:\n\t{a}")

def main():
    rank, world_size, local_rank, device = setup_distributed()
    info_and_test(rank, local_rank, world_size, device)

    # Clean up
    dist.destroy_process_group()

if __name__ == "__main__":
    main()

