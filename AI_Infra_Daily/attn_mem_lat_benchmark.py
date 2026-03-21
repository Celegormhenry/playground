import torch
from naive_attention import NaiveAttention
'''
 Latency:                                                                                                                                      
  - torch.cuda.Event(enable_timing=True) — create start/end events                                                                              
  - event.record() — mark a timestamp                                                                                                           
  - torch.cuda.synchronize() — wait for GPU to finish       
  - start.elapsed_time(end) — returns milliseconds between two events                                                                           
                                                                                                                                                
  Memory:                                                                                                                                       
  - torch.cuda.reset_peak_memory_stats() — reset counter before measuring                                                                       
  - torch.cuda.synchronize() — ensure all ops complete                                                                                          
  - torch.cuda.max_memory_allocated() — peak bytes used (divide by 1024**2 for MB)
'''

def measure_latency(fn, *args, warmup=5, repeats=10) -> float:
    """Returns average latency in milliseconds."""
    total_time = 0
    for i in range(warmup):
        output = fn(*args)  

    for j in range(repeats):
        start = torch.cuda.Event(enable_timing=True)                                                                                                  
        end = torch.cuda.Event(enable_timing=True)  
        start.record()                                                                                                                                
        output = fn(*args)                                                                                                                        
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)  
        total_time += ms
    return total_time / repeats      



def measure_memory(fn, *args) -> float:
    """Returns peak GPU memory in MB."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    output = fn(*args)
    torch.cuda.synchronize()                                                                                                                      
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) 
    return peak_mb

def run_test(seq_len:int):
    attn = NaiveAttention()
    SL = seq_len
    B, H, d = 2, 4, 64
    Q = torch.randn(B, H, SL, d, device='cuda')
    K = torch.randn(B, H, SL, d, device='cuda')
    V = torch.randn(B, H, SL, d, device='cuda')
    lat = measure_latency(attn, Q, K, V)
    mem = measure_memory(attn, Q, K, V)
    print(f"seq_len={SL}: latency={lat:.2f}ms, memory={mem:.2f}MB")


if __name__ == "__main__":
    for sl in [128, 256, 512, 1024, 2048, 4096]:
        run_test(sl)