import json,sys
rows=[]
for p in sys.argv[1:]:
    d=json.load(open(p)); rows.append((p.split("/")[0], list(d.values())[0]))
keys=["total_time","tensor_engine_active_time","tensor_engine_active_time_percent","vector_engine_active_time","scalar_engine_active_time","gpsimd_engine_active_time","dma_active_time","cc_op_active_time","cc_op_count","hbm_read_bytes","hbm_write_bytes","spill_save_bytes","spill_reload_bytes","hardware_flops","transpose_flops","mfu_estimated_percent","mbu_estimated_percent","throttle_active_nc4_time_ns","throttle_active_nc5_time_ns","throttle_avg_util_limit_nc4_percent","throttle_avg_util_limit_nc5_percent","throttle_activity_1_active_time_nc4_percent","matmul_instruction_count","vector_engine_instruction_count","scalar_engine_instruction_count","inputs_and_weights_size_bytes"]
print(f"{'metric':45s}"+"".join(f"{n:>22s}" for n,_ in rows))
for k in keys:
    print(f"{k:45s}"+"".join(f"{float(m.get(k) or 0):22.4g}" for _,m in rows))
# derived
for n,m in rows:
    t=float(m['total_time']); hf=float(m['hardware_flops']); tf=float(m.get('transpose_flops') or 0)
    print(n, f"HW TFLOPS/core {hf/t/1e12:.1f}  non-transpose {(hf-tf)/t/1e12:.1f}  HBM GB/s {(float(m['hbm_read_bytes'])+float(m['hbm_write_bytes']))/t/1e9:.0f}")
