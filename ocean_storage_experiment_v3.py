# -*- coding: utf-8 -*-
"""
《海洋大数据》课程大作业实验脚本（最终定稿版）
切入点：Parquet/Iceberg 湖仓布局下“单站长时序点查 (Q1)” vs “整片海面状查询 (Q2)”的读放大问题

定稿特性：
1. 显式 --out-dir 与 --overwrite 参数保护；使用 np.random.default_rng(42) 局部生成器。
2. Q1 显式补全 event_date 日期范围谓词，并加入空测站返回同 Schema 空表的边界保护。
3. 轻量路由表仅覆盖冷区，且包含 (min_date, max_date) 列，支持 Q1 精确路径的冷区文件级日期剪枝
   （仅查热区、单日冷区或无匹配日期时直接跳过无关冷文件）；并附短日期窗自检。
4. 补入同条件冷区消融对照（同一 10 天冷块，固定次级排序 (station_id, obs_time)）：
   对比 4 阶等步长 Hilbert、6 阶等步长 Hilbert 与 4 阶分位点 Hilbert。
5. 统一采用二进制单位 MiB / KiB / MiB/s；图 2 与图 3 柱顶标注“百分比 + (候选RG/总RG, MiB)”；
   预热后乱序重复查询 7 次取中位数（median），并打印软硬件环境与 Parquet 实际编码。
"""

import argparse
import glob
import json
import os
import platform
import shutil
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

# =========================
# 0. 全局基准参数
# =========================
N_DAYS = 100
N_STATIONS = 1000
ROWS_PER_STATION_DAY = 100
TOTAL_ROWS = N_DAYS * N_STATIONS * ROWS_PER_STATION_DAY  # 10,000,000 行
RG_SIZE = 10000  # 每个 Row Group 1 万行 (全表共 1,000 个 RG)

L_OPEN_MS = 20.0
BANDWIDTH_MIB_S = 500.0
MIB = 1024 * 1024
KIB = 1024
N_REPEATS = 7


def print_environment():
    print("=== 0. 运行环境信息 ===")
    print(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"CPU Cores (logical): {os.cpu_count()}")
    print(f"Python: {platform.python_version()} | PyArrow: {pa.__version__} | Pandas: {pd.__version__} | NumPy: {np.__version__}\n")


# =========================
# 1. Hilbert 编码与冻结分位点映射
# =========================
def xy2hilbert(n: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """向量化 2D -> 1D Hilbert 曲线编码"""
    d = np.zeros_like(x, dtype=np.int64)
    s = n // 2
    x = x.copy()
    y = y.copy()
    while s > 0:
        rx = ((x & s) > 0).astype(np.int64)
        ry = ((y & s) > 0).astype(np.int64)
        d += s * s * ((3 * rx) ^ ry)
        mask0 = (ry == 0)
        mask1 = mask0 & (rx == 1)
        x[mask1] = n - 1 - x[mask1]
        y[mask1] = n - 1 - y[mask1]
        tmp = x[mask0].copy()
        x[mask0] = y[mask0]
        y[mask0] = tmp
        s //= 2
    return d


def build_station_network(rng: np.random.Generator):
    """生成 1,000 个测站：80% 近岸密集 + 20% 远海稀疏"""
    n_nearshore = 800
    n_offshore = 200
    near_lat = rng.uniform(29.5, 31.5, n_nearshore)
    near_lon = rng.uniform(121.5, 123.5, n_nearshore)
    off_lat = rng.uniform(20.0, 35.0, n_offshore)
    off_lon = rng.uniform(123.5, 135.0, n_offshore)

    station_lats = np.concatenate([near_lat, off_lat])
    station_lons = np.concatenate([near_lon, off_lon])
    station_names = np.array([f"ST_{i:04d}" for i in range(N_STATIONS)])
    return station_names, station_lats, station_lons


def build_frozen_cdf_tables(station_lats: np.ndarray, station_lons: np.ndarray, cdf_path: str):
    """构建并持久化冻结静态空间 CDF 分位点查找表 (257 个分位点)"""
    lat_quantiles = np.quantile(station_lats, np.linspace(0, 1, 257))
    lon_quantiles = np.quantile(station_lons, np.linspace(0, 1, 257))
    with open(cdf_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_bins": 256,
                "lat_quantiles": lat_quantiles.tolist(),
                "lon_quantiles": lon_quantiles.tolist(),
            },
            f,
        )
    return lat_quantiles, lon_quantiles


def compute_quantile_hilbert(lats, lons, lat_q, lon_q, order=4):
    """O(log M) 查冻结分位点表后编码 Hilbert 桶 (order=4 -> 256 桶)"""
    n_grid = 1 << order
    qx = np.searchsorted(lon_q[1:-1], lons, side="right") / 256.0
    qy = np.searchsorted(lat_q[1:-1], lats, side="right") / 256.0
    ix = np.clip((qx * n_grid).astype(np.int64), 0, n_grid - 1)
    iy = np.clip((qy * n_grid).astype(np.int64), 0, n_grid - 1)
    return xy2hilbert(n_grid, ix, iy)


def compute_uniform_hilbert(lats, lons, order=6):
    """传统等步长 Hilbert 编码 (固定海域边界 lat:[20,35], lon:[121.5,135])"""
    n_grid = 1 << order
    nx = (lons - 121.5) / (135.0 - 121.5)
    ny = (lats - 20.0) / (35.0 - 20.0)
    ix = np.clip((nx * n_grid).astype(np.int64), 0, n_grid - 1)
    iy = np.clip((ny * n_grid).astype(np.int64), 0, n_grid - 1)
    return xy2hilbert(n_grid, ix, iy)


# =========================
# 2. 数据生成与 4 种物理布局 + 消融实验写入
# =========================
CORE_SCHEMA = pa.schema([
    ("station_id", pa.string()),
    ("obs_time", pa.timestamp("us")),
    ("event_date", pa.string()),
    ("ingest_time", pa.timestamp("us")),
    ("lat", pa.float64()),
    ("lon", pa.float64()),
    ("sst", pa.float64()),
])


def prepare_output_dir(out_dir: str, overwrite: bool):
    if os.path.exists(out_dir) and os.listdir(out_dir):
        if not overwrite:
            raise FileExistsError(
                f"输出目录 '{out_dir}' 已存在且非空。为防止误删数据，请使用 --overwrite 或指定新的 --out-dir。"
            )
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)


def generate_and_write_layouts(out_dir: str, rng: np.random.Generator):
    station_names, station_lats, station_lons = build_station_network(rng)
    cdf_path = os.path.join(out_dir, "frozen_cdf_table.json")
    lat_q, lon_q = build_frozen_cdf_tables(station_lats, station_lons, cdf_path)

    station_uni_h4 = compute_uniform_hilbert(station_lats, station_lons, order=4)
    station_uni_h6 = compute_uniform_hilbert(station_lats, station_lons, order=6)
    station_q_h4 = compute_quantile_hilbert(station_lats, station_lons, lat_q, lon_q, order=4)

    layout_dirs = {
        "L1_time": os.path.join(out_dir, "L1_time"),
        "L2_station": os.path.join(out_dir, "L2_station"),
        "L3_uni_hilbert": os.path.join(out_dir, "L3_uni_hilbert"),
        "L4_ours": os.path.join(out_dir, "L4_ours"),
    }
    ablation_dir = os.path.join(out_dir, "ablation_cold_chunk07")
    for d in list(layout_dirs.values()) + [ablation_dir]:
        os.makedirs(d, exist_ok=True)

    start_date = pd.Timestamp("2026-01-01")
    base_st_idx = np.repeat(np.arange(N_STATIONS), ROWS_PER_STATION_DAY)

    write_times = {k: 0.0 for k in layout_dirs}
    cold_routing_records = []
    cold_buffer = []
    ablation_chunk7_frames = []

    print(f"=== 1. 生成 {TOTAL_ROWS:,} 行模拟数据并写入 4 种物理布局 (7 列统一 Schema) ===")
    t0 = time.perf_counter()

    for day in range(N_DAYS):
        cur_date = start_date + pd.Timedelta(days=day)
        date_str = cur_date.strftime("%Y-%m-%d")

        obs_sec = np.tile(np.linspace(0, 86399, ROWS_PER_STATION_DAY, dtype=np.int64), N_STATIONS)
        obs_sec += rng.integers(-30, 31, size=len(obs_sec))
        obs_sec = np.clip(obs_sec, 0, 86399)
        ingest_sec = obs_sec + rng.integers(1, 300, size=len(obs_sec))

        base_ts = cur_date.value // 1000
        obs_ts = base_ts + obs_sec * 1_000_000
        ingest_ts = base_ts + ingest_sec * 1_000_000

        lats = station_lats[base_st_idx]
        df_day = pd.DataFrame({
            "station_id": station_names[base_st_idx],
            "obs_time": pd.to_datetime(obs_ts, unit="us"),
            "event_date": date_str,
            "ingest_time": pd.to_datetime(ingest_ts, unit="us"),
            "lat": lats,
            "lon": station_lons[base_st_idx],
            "sst": 15.0 + 0.3 * (30.0 - lats) + rng.normal(0, 0.5, size=len(obs_sec)),
            "_q_h4": station_q_h4[base_st_idx],
            "_uni_h4": station_uni_h4[base_st_idx],
            "_uni_h6": station_uni_h6[base_st_idx],
        })

        if 70 <= day < 80:
            ablation_chunk7_frames.append(df_day.copy())

        core_cols = CORE_SCHEMA.names

        # L1: 按天 + ingest_time
        tw = time.perf_counter()
        df_l1 = df_day.sort_values("ingest_time", kind="mergesort")[core_cols]
        pq.write_table(
            pa.Table.from_pandas(df_l1, schema=CORE_SCHEMA, preserve_index=False),
            os.path.join(layout_dirs["L1_time"], f"day_{day:03d}.parquet"),
            row_group_size=RG_SIZE, compression="snappy",
        )
        write_times["L1_time"] += time.perf_counter() - tw

        # L2: 按天 + (station_id, obs_time)
        tw = time.perf_counter()
        df_l2 = df_day.sort_values(["station_id", "obs_time"], kind="mergesort")[core_cols]
        pq.write_table(
            pa.Table.from_pandas(df_l2, schema=CORE_SCHEMA, preserve_index=False),
            os.path.join(layout_dirs["L2_station"], f"day_{day:03d}.parquet"),
            row_group_size=RG_SIZE, compression="snappy",
        )
        write_times["L2_station"] += time.perf_counter() - tw

        # L3: 按天 + 传统等步长 6 阶 Hilbert
        tw = time.perf_counter()
        df_l3 = df_day.sort_values(["_uni_h6", "obs_time"], kind="mergesort")[core_cols]
        pq.write_table(
            pa.Table.from_pandas(df_l3, schema=CORE_SCHEMA, preserve_index=False),
            os.path.join(layout_dirs["L3_uni_hilbert"], f"day_{day:03d}.parquet"),
            row_group_size=RG_SIZE, compression="snappy",
        )
        write_times["L3_uni_hilbert"] += time.perf_counter() - tw

        # L4: 冷区 (前90天每10天合并) 按 (_q_h4, station_id, obs_time) 排序；
        #     热区 (近10天按天) 按 ingest_time 排序；冷区路由表记录 min_date, max_date
        tw = time.perf_counter()
        if day < 90:
            cold_buffer.append(df_day)
            if (day + 1) % 10 == 0:
                chunk_id = day // 10
                df_merged = pd.concat(cold_buffer, ignore_index=True)
                df_merged = df_merged.sort_values(["_q_h4", "station_id", "obs_time"], kind="mergesort")
                fname = f"cold_chunk_{chunk_id:02d}.parquet"
                fpath = os.path.join(layout_dirs["L4_ours"], fname)
                pq.write_table(
                    pa.Table.from_pandas(df_merged[core_cols], schema=CORE_SCHEMA, preserve_index=False),
                    fpath, row_group_size=RG_SIZE, compression="snappy",
                )
                c_min_date = str(df_merged["event_date"].min())
                c_max_date = str(df_merged["event_date"].max())
                n_rgs = (len(df_merged) + RG_SIZE - 1) // RG_SIZE
                st_arr = df_merged["station_id"].values
                for rg_id in range(n_rgs):
                    for st in np.unique(st_arr[rg_id * RG_SIZE:min((rg_id + 1) * RG_SIZE, len(df_merged))]):
                        cold_routing_records.append((st, fname, rg_id, c_min_date, c_max_date))
                cold_buffer.clear()
        else:
            df_hot = df_day.sort_values("ingest_time", kind="mergesort")[core_cols]
            fname = f"hot_day_{day:03d}.parquet"
            fpath = os.path.join(layout_dirs["L4_ours"], fname)
            pq.write_table(
                pa.Table.from_pandas(df_hot, schema=CORE_SCHEMA, preserve_index=False),
                fpath, row_group_size=RG_SIZE, compression="snappy",
            )
        write_times["L4_ours"] += time.perf_counter() - tw

    # 写入含 min_date, max_date 的冷区轻量级 RG 路由表
    tw = time.perf_counter()
    df_route = pd.DataFrame(
        cold_routing_records,
        columns=["station_id", "file_name", "rg_id", "min_date", "max_date"],
    )
    df_route = df_route.sort_values(["station_id", "file_name", "rg_id"])
    route_path = os.path.join(out_dir, "cold_routing_table.parquet")
    pq.write_table(pa.Table.from_pandas(df_route, preserve_index=False), route_path, compression="snappy")
    write_times["L4_ours"] += time.perf_counter() - tw

    # 写入冷区同条件消融对照文件 (固定 10 天冷块 + 次级排序 (station_id, obs_time))
    df_abl = pd.concat(ablation_chunk7_frames, ignore_index=True)
    ablation_files = {}
    for key, sort_col in [("uni_h4", "_uni_h4"), ("uni_h6", "_uni_h6"), ("q_h4", "_q_h4")]:
        df_sorted = df_abl.sort_values([sort_col, "station_id", "obs_time"], kind="mergesort")[CORE_SCHEMA.names]
        apath = os.path.join(ablation_dir, f"cold_chunk_07_{key}.parquet")
        pq.write_table(
            pa.Table.from_pandas(df_sorted, schema=CORE_SCHEMA, preserve_index=False),
            apath, row_group_size=RG_SIZE, compression="snappy",
        )
        ablation_files[key] = apath

    print(f"写入完成，总耗时: {time.perf_counter() - t0:.2f}s\n")
    print("=== 2. 物理存储与写入耗时统计 (二进制单位 MiB / KiB) ===")
    for name, d in layout_dirs.items():
        files = sorted(glob.glob(os.path.join(d, "*.parquet")))
        total_bytes = sum(os.path.getsize(f) for f in files)
        total_rgs = sum(pq.ParquetFile(f).metadata.num_row_groups for f in files)
        print(f"{name:15s} | 写入耗时: {write_times[name]:6.2f}s | 文件数: {len(files):3d} | 总RG数: {total_rgs:4d} | 总大小: {total_bytes/MIB:6.2f} MiB")

    route_kib = os.path.getsize(route_path) / KIB
    cdf_kib = os.path.getsize(cdf_path) / KIB
    print(f"冷区RG路由表    | 记录数: {len(df_route):7,d} 条 | 文件大小: {route_kib:.2f} KiB (含 min_date/max_date)")
    print(f"冻结CDF查找表   | 分位点:     257 个 | 文件大小: {cdf_kib:.2f} KiB")

    sample_pf = pq.ParquetFile(sorted(glob.glob(os.path.join(layout_dirs["L4_ours"], "cold_*.parquet")))[0])
    rg0 = sample_pf.metadata.row_group(0)
    print("\n[Parquet 物理编码抽检 (L4冷区 RG0)]")
    for ci in range(rg0.num_columns):
        col_meta = rg0.column(ci)
        print(f"  列 {col_meta.path_in_schema:12s} | 编码: {col_meta.encodings}")
    print()

    return layout_dirs, route_path, ablation_files


# =========================
# 3. 剪枝统计、乱序 7 次中位数计时与对象存储仿真
# =========================
def estimate_footer_pruning_bytes(files, filters, cols_to_read):
    """基于 Parquet Footer Min-Max 统计估算候选文件数、RG 数与候选列块压缩量 (Bytes)"""
    files_hit = 0
    rgs_read = 0
    rgs_in_hit_files = 0
    total_rgs = 0
    bytes_read = 0
    for f in files:
        pf = pq.ParquetFile(f)
        meta = pf.metadata
        col_idx_map = {pf.schema_arrow.names[i]: i for i in range(meta.num_columns)}
        file_has_match = False
        for rg_i in range(meta.num_row_groups):
            total_rgs += 1
            rg_meta = meta.row_group(rg_i)
            match = True
            for col, (op, val) in filters.items():
                stats = rg_meta.column(col_idx_map[col]).statistics
                if stats is not None and stats.has_min_max:
                    cmin, cmax = stats.min, stats.max
                    if op == "==" and (val < cmin or val > cmax):
                        match = False
                        break
                    elif op == "between" and (val[1] < cmin or val[0] > cmax):
                        match = False
                        break
            if match:
                file_has_match = True
                rgs_read += 1
                for c in cols_to_read:
                    bytes_read += rg_meta.column(col_idx_map[c]).total_compressed_size
        if file_has_match:
            files_hit += 1
            rgs_in_hit_files += meta.num_row_groups
    return files_hit, rgs_read, rgs_in_hit_files, total_rgs, bytes_read


def eval_sim_time_ms(files_hit: int, bytes_read: int, extra_ms: float = 0.0) -> float:
    return files_hit * L_OPEN_MS + (bytes_read / (BANDWIDTH_MIB_S * MIB)) * 1000.0 + extra_ms


def query_l4_exact_q1(l4_dir: str, route_path: str, station_id: str, date_start: str, date_end: str, return_stats: bool = False):
    """L4 精确冷路由（带 min_date/max_date 区间剪枝）+ 热区 Dataset 扫描（带日期剪枝与空结果保护）"""
    q1_schema = pa.schema([("obs_time", pa.timestamp("us")), ("sst", pa.float64())])
    q1_cols = ["obs_time", "sst"]
    q1_read_cols = ["station_id", "event_date", "obs_time", "sst"]

    # 1. 冷区路由表先做 station_id + [date_start, date_end] 交集剪枝
    rt_tbl = pq.read_table(
        route_path,
        filters=[
            ("station_id", "==", station_id),
            ("max_date", ">=", date_start),
            ("min_date", "<=", date_end),
        ],
    )
    parts = []
    cold_files_opened = 0
    cold_rgs_read = 0
    if rt_tbl.num_rows > 0:
        grouped = rt_tbl.to_pandas().groupby("file_name")["rg_id"].apply(list).to_dict()
        cold_files_opened = len(grouped)
        for fname, rg_ids in grouped.items():
            cold_rgs_read += len(rg_ids)
            pf = pq.ParquetFile(os.path.join(l4_dir, fname))
            sub = pf.read_row_groups(rg_ids, columns=q1_read_cols)
            mask = pc.and_kleene(
                pc.equal(sub["station_id"], station_id),
                pc.and_kleene(
                    pc.greater_equal(sub["event_date"], date_start),
                    pc.less_equal(sub["event_date"], date_end),
                ),
            )
            filtered = sub.filter(mask).select(q1_cols)
            if filtered.num_rows > 0:
                parts.append(filtered)

    # 2. 热区直走 Dataset 谓词下推（若日期窗口不落入热区，Min-Max 剪枝自动跳过热区文件）
    hot_files = sorted(glob.glob(os.path.join(l4_dir, "hot_*.parquet")))
    if hot_files:
        hot_flt = (
            (ds.field("station_id") == station_id)
            & (ds.field("event_date") >= date_start)
            & (ds.field("event_date") <= date_end)
        )
        hot_tbl = ds.dataset(hot_files, format="parquet").to_table(columns=q1_cols, filter=hot_flt)
        if hot_tbl.num_rows > 0:
            parts.append(hot_tbl)

    out_tbl = pa.concat_tables(parts) if parts else pa.Table.from_pylist([], schema=q1_schema)
    if return_stats:
        return out_tbl, cold_files_opened, cold_rgs_read
    return out_tbl


def benchmark_median_shuffled(tasks: dict, rng: np.random.Generator, n_repeats: int = N_REPEATS):
    """预热后乱序执行 n_repeats 轮，返回各任务本地耗时中位数 (ms) 与结果行数"""
    timings = {k: [] for k in tasks}
    row_counts = {}

    for k, fn in tasks.items():
        tbl = fn()
        row_counts[k] = tbl.num_rows

    keys = list(tasks.keys())
    for _ in range(n_repeats):
        order = rng.permutation(len(keys))
        for idx in order:
            k = keys[idx]
            t1 = time.perf_counter()
            tbl = tasks[k]()
            timings[k].append((time.perf_counter() - t1) * 1000.0)
            row_counts[k] = tbl.num_rows

    return {k: float(np.median(v)) for k, v in timings.items()}, row_counts


def run_all_benchmarks(layout_dirs, route_path, ablation_files, rng: np.random.Generator):
    target_station = "ST_0100"
    d_start, d_end = "2026-01-01", "2026-04-10"
    q2_lat_min, q2_lat_max = 30.3, 30.6
    q2_lon_min, q2_lon_max = 122.3, 122.6

    # 短日期窗与空结果自检验证
    print("=== 3. L4 精确路由短日期窗与空结果自检 ===")
    chk_cases = [
        ("仅查热区单日", target_station, "2026-04-06", "2026-04-06"),
        ("仅查冷区单日", target_station, "2026-03-15", "2026-03-15"),
        ("越界无匹配日期", target_station, "2026-05-01", "2026-05-01"),
        ("不存在测站", "ST_9999", d_start, d_end),
    ]
    for desc, st, ds_s, ds_e in chk_cases:
        tbl, c_f, c_rg = query_l4_exact_q1(layout_dirs["L4_ours"], route_path, st, ds_s, ds_e, return_stats=True)
        print(f"  [{desc:10s}] ({st}, {ds_s}~{ds_e}) -> 命中冷文件:{c_f} | 读冷RG:{c_rg} | 返回行数:{tbl.num_rows}")
    print()

    results = {"Q1": [], "Q2_cold_day": [], "Q2_cold_hour": [], "Q2_hot_day": [], "Q2_hot_hour": [], "ablation": []}

    # ---------- Q1 任务注册与 Footer 统计 ----------
    q1_cols = ["obs_time", "sst"]
    q1_all_cols = ["station_id", "event_date", "obs_time", "sst"]
    q1_ds_flt = (
        (ds.field("station_id") == target_station)
        & (ds.field("event_date") >= d_start)
        & (ds.field("event_date") <= d_end)
    )
    q1_stat_flt = {"station_id": ("==", target_station), "event_date": ("between", (d_start, d_end))}

    q1_tasks, q1_meta = {}, {}
    for name, d in layout_dirs.items():
        files = sorted(glob.glob(os.path.join(d, "*.parquet")))
        f_hit, rgs_r, _, rgs_tot, b_read = estimate_footer_pruning_bytes(files, q1_stat_flt, q1_all_cols)
        tkey = f"{name}_P1" if name == "L4_ours" else name
        q1_tasks[tkey] = (lambda dir_path=d: ds.dataset(dir_path, format="parquet").to_table(columns=q1_cols, filter=q1_ds_flt))
        q1_meta[tkey] = (f_hit, len(files), rgs_r, rgs_tot, b_read, eval_sim_time_ms(f_hit, b_read))

        if name == "L4_ours":
            rt_df = pq.read_table(
                route_path,
                filters=[("station_id", "==", target_station), ("max_date", ">=", d_start), ("min_date", "<=", d_end)],
            ).to_pandas()
            cold_grouped = rt_df.groupby("file_name")["rg_id"].apply(list).to_dict()
            cold_b_read, cold_rgs_r = 0, 0
            for fname, rg_ids in cold_grouped.items():
                pf = pq.ParquetFile(os.path.join(d, fname))
                col_idx_map = {pf.schema_arrow.names[i]: i for i in range(pf.metadata.num_columns)}
                for rg_id in rg_ids:
                    cold_rgs_r += 1
                    rg_meta = pf.metadata.row_group(rg_id)
                    for c in q1_all_cols:
                        cold_b_read += rg_meta.column(col_idx_map[c]).total_compressed_size
            hot_files = sorted(glob.glob(os.path.join(d, "hot_*.parquet")))
            h_f_hit, h_rgs_r, _, _, h_b_read = estimate_footer_pruning_bytes(hot_files, q1_stat_flt, q1_all_cols)
            tot_f_hit = len(cold_grouped) + h_f_hit
            tot_rgs_r = cold_rgs_r + h_rgs_r
            tot_b_read = cold_b_read + h_b_read
            sim_ms = eval_sim_time_ms(tot_f_hit, tot_b_read, extra_ms=20.0)
            q1_tasks["L4_ours_P2"] = (lambda dir_path=d: query_l4_exact_q1(dir_path, route_path, target_station, d_start, d_end))
            q1_meta["L4_ours_P2"] = (tot_f_hit, len(files), tot_rgs_r, rgs_tot, tot_b_read, sim_ms)

    q1_medians, q1_rows = benchmark_median_shuffled(q1_tasks, rng)

    print("=== 4. 查询实测（预热后乱序 7 次中位数）与 Footer 估算对比 (L_open=20ms, B=500MiB/s) ===")
    print(f"\n[Q1] 单近岸站 ({target_station}) 100天时序")
    for tkey in ["L1_time", "L2_station", "L3_uni_hilbert", "L4_ours_P1", "L4_ours_P2"]:
        f_hit, f_tot, rgs_r, rgs_tot, b_read, sim_ms = q1_meta[tkey]
        med_ms = q1_medians[tkey]
        mib = b_read / MIB
        results["Q1"].append((tkey, mib, sim_ms, med_ms, rgs_r, f_hit))
        print(f"  {tkey:16s} | 行数:{q1_rows[tkey]:6d} | 命中文件:{f_hit:3d}/{f_tot:3d} | 候选RG:{rgs_r:4d}/{rgs_tot:4d} | Footer估算列块:{mib:6.2f} MiB | 本地7次中位:{med_ms:6.2f} ms | 对象存储仿真:{sim_ms:7.1f} ms")

    # ---------- Q2 冷热双区对比 ----------
    q2_configs = [
        ("Q2_cold", "2026-03-15", pd.Timestamp("2026-03-15 10:00:00"), pd.Timestamp("2026-03-15 11:00:00")),
        ("Q2_hot",  "2026-04-06", pd.Timestamp("2026-04-06 10:00:00"), pd.Timestamp("2026-04-06 11:00:00")),
    ]
    q2_cols = ["station_id", "lat", "lon", "sst"]
    q2_day_cols = ["event_date", "station_id", "lat", "lon", "sst"]
    q2_hr_cols = ["event_date", "obs_time", "station_id", "lat", "lon", "sst"]

    for lbl, t_date, h_s, h_e in q2_configs:
        day_stat = {"event_date": ("==", t_date), "lat": ("between", (q2_lat_min, q2_lat_max)), "lon": ("between", (q2_lon_min, q2_lon_max))}
        day_flt = ((ds.field("event_date") == t_date) & (ds.field("lat") >= q2_lat_min) & (ds.field("lat") <= q2_lat_max) &
                   (ds.field("lon") >= q2_lon_min) & (ds.field("lon") <= q2_lon_max))
        d_tasks, d_meta = {}, {}
        for name, d in layout_dirs.items():
            files = sorted(glob.glob(os.path.join(d, "*.parquet")))
            f_hit, rgs_r, rgs_hit_f, _, b_read = estimate_footer_pruning_bytes(files, day_stat, q2_day_cols)
            d_tasks[name] = (lambda dir_path=d: ds.dataset(dir_path, format="parquet").to_table(columns=q2_cols, filter=day_flt))
            d_meta[name] = (f_hit, len(files), rgs_r, rgs_hit_f, b_read, eval_sim_time_ms(f_hit, b_read))
        d_med, d_rows = benchmark_median_shuffled(d_tasks, rng)
        print(f"\n[{lbl}_day] 单日近岸 0.3°x0.3° 快照 ({t_date})")
        for name in layout_dirs:
            f_hit, f_tot, rgs_r, rgs_hit_f, b_read, sim_ms = d_meta[name]
            skip_pct = (1.0 - rgs_r / rgs_hit_f) * 100.0 if rgs_hit_f > 0 else 0.0
            results[f"{lbl}_day"].append((name, skip_pct, b_read / MIB, rgs_r, rgs_hit_f))
            print(f"  {name:15s} | 行数:{d_rows[name]:5d} | 命中:{f_hit:2d}/{f_tot:3d} | 候选RG:{rgs_r:3d}/{rgs_hit_f:3d}RG (跳过:{skip_pct:5.1f}%) | Footer估算:{b_read/MIB:5.2f} MiB | 本地7次中位:{d_med[name]:6.2f} ms | 对象存储:{sim_ms:6.1f} ms")

        hr_stat = {"event_date": ("==", t_date), "obs_time": ("between", (h_s.to_pydatetime(), h_e.to_pydatetime())),
                   "lat": ("between", (q2_lat_min, q2_lat_max)), "lon": ("between", (q2_lon_min, q2_lon_max))}
        hr_flt = ((ds.field("event_date") == t_date) &
                  (ds.field("obs_time") >= pa.scalar(h_s.to_pydatetime(), type=pa.timestamp("us"))) &
                  (ds.field("obs_time") <= pa.scalar(h_e.to_pydatetime(), type=pa.timestamp("us"))) &
                  (ds.field("lat") >= q2_lat_min) & (ds.field("lat") <= q2_lat_max) &
                  (ds.field("lon") >= q2_lon_min) & (ds.field("lon") <= q2_lon_max))
        h_tasks, h_meta = {}, {}
        for name, d in layout_dirs.items():
            files = sorted(glob.glob(os.path.join(d, "*.parquet")))
            f_hit, rgs_r, rgs_hit_f, _, b_read = estimate_footer_pruning_bytes(files, hr_stat, q2_hr_cols)
            h_tasks[name] = (lambda dir_path=d: ds.dataset(dir_path, format="parquet").to_table(columns=q2_cols, filter=hr_flt))
            h_meta[name] = (f_hit, len(files), rgs_r, rgs_hit_f, b_read, eval_sim_time_ms(f_hit, b_read))
        h_med, h_rows = benchmark_median_shuffled(h_tasks, rng)
        print(f"\n[{lbl}_hour] 1小时近岸 0.3°x0.3° 快照 ({h_s} ~ {h_e})")
        for name in layout_dirs:
            f_hit, f_tot, rgs_r, rgs_hit_f, b_read, sim_ms = h_meta[name]
            skip_pct = (1.0 - rgs_r / rgs_hit_f) * 100.0 if rgs_hit_f > 0 else 0.0
            results[f"{lbl}_hour"].append((name, skip_pct, b_read / MIB, rgs_r, rgs_hit_f))
            print(f"  {name:15s} | 行数:{h_rows[name]:5d} | 命中:{f_hit:2d}/{f_tot:3d} | 候选RG:{rgs_r:3d}/{rgs_hit_f:3d}RG (跳过:{skip_pct:5.1f}%) | Footer估算:{b_read/MIB:5.2f} MiB | 本地7次中位:{h_med[name]:6.2f} ms | 对象存储:{sim_ms:6.1f} ms")

    # ---------- 冷区同条件消融对照 ----------
    print("\n[冷区同条件消融对照] 同一 10 天冷块 (100万行/100 RG), 固定次级排序 (station_id, obs_time), 测试 Q2_cold_day (2026-03-15)")
    abl_stat = {"event_date": ("==", "2026-03-15"), "lat": ("between", (q2_lat_min, q2_lat_max)), "lon": ("between", (q2_lon_min, q2_lon_max))}
    abl_flt = ((ds.field("event_date") == "2026-03-15") & (ds.field("lat") >= q2_lat_min) & (ds.field("lat") <= q2_lat_max) &
               (ds.field("lon") >= q2_lon_min) & (ds.field("lon") <= q2_lon_max))
    abl_tasks, abl_meta = {}, {}
    for key, apath in ablation_files.items():
        _, rgs_r, rgs_hit_f, _, b_read = estimate_footer_pruning_bytes([apath], abl_stat, q2_day_cols)
        abl_tasks[key] = (lambda p=apath: ds.dataset(p, format="parquet").to_table(columns=q2_cols, filter=abl_flt))
        abl_meta[key] = (rgs_r, rgs_hit_f, b_read)
    abl_med, abl_rows = benchmark_median_shuffled(abl_tasks, rng)
    for key in ["uni_h4", "uni_h6", "q_h4"]:
        rgs_r, rgs_hit_f, b_read = abl_meta[key]
        skip_pct = (1.0 - rgs_r / rgs_hit_f) * 100.0
        results["ablation"].append((key, skip_pct, b_read / MIB, rgs_r, rgs_hit_f))
        print(f"  {key:10s} | 行数:{abl_rows[key]:5d} | 候选RG:{rgs_r:3d}/{rgs_hit_f:3d}RG (跳过:{skip_pct:5.1f}%) | Footer估算:{b_read/MIB:5.2f} MiB | 本地7次中位:{abl_med[key]:6.2f} ms")

    return results


# =========================
# 4. Matplotlib 实验结果对比绘图 (MiB / MiB/s + 完整标签)
# =========================
def plot_benchmark_charts(results, plot_dir: str):
    import matplotlib.pyplot as plt
    os.makedirs(plot_dir, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 图 1: Q1 Footer 估算列压缩读取量 (MiB) 与对象存储仿真耗时 (ms)
    q1_labels = [r[0] for r in results["Q1"]]
    q1_mib = [r[1] for r in results["Q1"]]
    q1_sim = [r[2] for r in results["Q1"]]
    colors = ["#7f8c8d", "#3498db", "#e67e22", "#9b59b6", "#2ecc71"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    b1 = ax1.bar(q1_labels, q1_mib, color=colors)
    ax1.set_title("Q1 (单站100天时序) Footer 估算候选列块压缩量 (MiB)")
    ax1.set_ylabel("列压缩读取量 (MiB)")
    ax1.tick_params(axis="x", rotation=18)
    for b in b1:
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"{b.get_height():.2f}", ha="center", fontsize=9)

    b2 = ax2.bar(q1_labels, q1_sim, color=colors)
    ax2.set_title("Q1 对象存储仿真耗时 (ms, L_open=20ms, B=500MiB/s)")
    ax2.set_ylabel("仿真耗时 (ms)")
    ax2.tick_params(axis="x", rotation=18)
    for b in b2:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 30, f"{b.get_height():.1f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "fig1_q1_mib_and_sim_latency.png"), dpi=160)
    plt.close()

    # 图 2: Q2 冷区单日 vs 热区 1 小时文件内 RG 跳过率 (%) + (候选RG/总RG, MiB)
    layouts = [r[0] for r in results["Q2_cold_day"]]
    cold_rows = results["Q2_cold_day"]
    hot_rows = results["Q2_hot_hour"]
    x = np.arange(len(layouts))
    w = 0.38

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    r1 = ax.bar(x - w / 2, [r[1] for r in cold_rows], w, label="Q2_cold_day (冷区单日 0.3°×0.3°)", color="#2980b9")
    r2 = ax.bar(x + w / 2, [r[1] for r in hot_rows], w, label="Q2_hot_hour (热区1小时 0.3°×0.3°)", color="#27ae60")
    ax.set_xticks(x)
    ax.set_xticklabels(layouts)
    ax.set_ylim(0, 122)
    ax.set_ylabel("命中文件内 Row Group 跳过率 (%)")
    ax.set_title("Q2 近岸小海区查询：冷区空间剪枝 vs 热区时间剪枝")
    ax.legend(loc="upper left")
    for b, row in zip(r1, cold_rows):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.8,
                f"{row[1]:.1f}%\n({row[3]}/{row[4]}, {row[2]:.2f}MiB)", ha="center", va="bottom", fontsize=7.8)
    for b, row in zip(r2, hot_rows):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.8,
                f"{row[1]:.1f}%\n({row[3]}/{row[4]}, {row[2]:.2f}MiB)", ha="center", va="bottom", fontsize=7.8)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "fig2_q2_cold_vs_hot_skip_rate.png"), dpi=160)
    plt.close()

    # 图 3: 冷区同条件消融对照 + (候选RG/总RG, MiB)
    abl_labels = ["4阶等步长\nHilbert", "6阶等步长\nHilbert", "4阶分位点\nHilbert (ours)"]
    abl_rows = results["ablation"]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bars = ax.bar(abl_labels, [r[1] for r in abl_rows], color=["#95a5a6", "#e67e22", "#2ecc71"], width=0.48)
    ax.set_ylim(0, 118)
    ax.set_ylabel("冷区 10 天文件内 RG 跳过率 (%)")
    ax.set_title("冷区同条件消融对照 (固定次级排序: station_id, obs_time)")
    for b, row in zip(bars, abl_rows):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.8,
                f"{row[1]:.1f}%\n({row[3]}/{row[4]}, {row[2]:.2f}MiB)", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "fig3_cold_ablation_skip_rate.png"), dpi=160)
    plt.close()
    print(f"\n图表已保存至 {plot_dir}/")


def main():
    parser = argparse.ArgumentParser(description="海洋测站 Parquet 布局双向读放大对比实验")
    parser.add_argument("--out-dir", default="./exp_ocean_data", help="实验数据输出目录 (默认: ./exp_ocean_data)")
    parser.add_argument("--plot-dir", default="./exp_plots", help="Matplotlib 图表输出目录 (默认: ./exp_plots)")
    parser.add_argument("--overwrite", action="store_true", help="若输出目录非空，允许清空并覆盖写入")
    args = parser.parse_args()

    print_environment()
    rng = np.random.default_rng(42)
    prepare_output_dir(args.out_dir, args.overwrite)
    dirs, r_path, abl_files = generate_and_write_layouts(args.out_dir, rng)
    res = run_all_benchmarks(dirs, r_path, abl_files, rng)
    plot_benchmark_charts(res, args.plot_dir)


if __name__ == "__main__":
    main()