import sys, duckdb, numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60); pd.set_option("display.max_rows", 100)
pq = sys.argv[1]
con = duckdb.connect()
con.sql(f"create view I as select *, coalesce(nullif(regexp_replace(nki_source_location, '.*/site-packages/|.*/Difflet/', ''),'nan'),'<none>') src from '{pq}/Instruction.parquet'")
span = con.sql("select (max(end_ts)-min(start_ts))/1e6 from I").fetchone()[0]
print(f"span {span:.1f} ms")
print("== TensorE by type (both pcores)")
print(con.sql("select tensor_instruction_type t, opcode, count(*) n, round(sum(duration_ns)/1e6,1) ms, round(sum(raw_flops)/1e12,2) tflop from I where engine='Tensor' and opcode in ('MATMUL','LDWEIGHTS') group by all order by ms desc").df().to_string())
def cat(s):
    if 'attention_cte' in s: return 'attention(NKI)'
    if 'layers.py' in s or 'module.py:1786' in s: return 'linear'
    if 'normalization.py' in s: return 'layernorm'
    if 'modeling_wan.py:52' in s or 'modeling_wan.py:49' in s: return 'qk-rmsnorm'
    if 'embeddings.py' in s: return 'rope'
    if s == '<none>': return 'compiler(no src)'
    return 'other'
df = con.sql("select src, engine, tensor_instruction_type t, sum(duration_ns)/1e6 ms from I group by all").df()
df['cat'] = df.src.map(cat)
df.loc[(df.engine=='Tensor') & (df.t=='TRANSPOSE'), 'cat'] = 'transpose(PE)'
pv = df.pivot_table(index='cat', columns='engine', values='ms', aggfunc='sum', fill_value=0)
pv['total'] = pv.sum(axis=1)
print("== summed instruction time by category x engine (ms, both pcores)")
print(pv.sort_values('total', ascending=False).round(1).to_string())
print("== top sources")
pv2 = df.pivot_table(index='src', columns='engine', values='ms', aggfunc='sum', fill_value=0); pv2['total']=pv2.sum(axis=1)
print(pv2.sort_values('total', ascending=False).head(14).round(1).to_string())
print("== CC ops")
print(con.sql(f"""select operation, dtype, input_size, count(*) n, round(sum(duration_ns)/1e6,2) ms, round(avg(duration_ns)/1e3,1) avg_us from '{pq}/CcOp.parquet' group by all order by ms desc""").df().to_string())
print(con.sql(f"select type, pcore_idx, k, n, count(*) cnt, round(sum(duration)/1e6,1) ms from '{pq}/Throttle.parquet' group by all order by all").df().to_string())
# idle analysis
at = con.sql(f"select engine, pcore_idx, start_ts, end_ts from '{pq}/ActiveTime.parquet'").df()
cc = con.sql(f"select start_ts, end_ts from '{pq}/CcOp.parquet'").df()
T0, T1 = int(at.start_ts.min()), int(at.end_ts.max()); N = (T1-T0)//1000+1
def mask(d):
    m = np.zeros(N, bool)
    for s, e in zip((d.start_ts.values-T0)//1000, (d.end_ts.values-T0)//1000): m[s:e+1] = True
    return m
ccm = mask(cc)
for p in (0, 1):
    d = at[at.pcore_idx == p]; eng = {e: mask(d[d.engine == e]) for e in d.engine.unique()}
    t = eng['tensor']; idle = ~t; other = np.zeros(N, bool)
    for k, v in eng.items():
        if k != 'tensor': other |= v
    vs = eng.get('vector', np.zeros(N,bool)) | eng.get('scalar', np.zeros(N,bool))
    print(f"pcore {p}: span {N/1000:.1f} ms | tensor busy {t.sum()/1000:.1f} | idle&vec/scal {np.sum(idle&vs)/1000:.1f} | idle&only-cc {np.sum(idle&ccm&~other)/1000:.1f} | idle&nothing {np.sum(idle&~other&~ccm)/1000:.1f}")
print("== DMA by queue type")
print(con.sql(f"select queue_type, round(sum(read_bytes)/1e9,2) rd_GB, round(sum(duration_ns)/1e6,1) ms_sum from '{pq}/DmaPacket.parquet' group by all order by rd_GB desc").df().to_string())
print(con.sql(f"select pcore_idx, queue_name, round(sum(read_bytes)/1e9,2) rd_GB, round(sum(case when is_transpose_mode then transfer_bytes else 0 end)/1e9,2) transpose_GB, round(sum(duration_ns)/1e6,1) ms from '{pq}/DmaPacketAggregated.parquet' group by all order by rd_GB desc limit 6").df().to_string())
print(con.sql(f"select * from '{pq}/HbmUsageSummaryByType.parquet' where usage_type in ('DMA Rings Spill','Tensors','Shared Scratchpad','Tensor Instructions','Total') order by neuroncore_idx").df().to_string())
