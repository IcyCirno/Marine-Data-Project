# -*- coding: utf-8 -*-
"""
《海洋大数据》课程大作业实验脚本
切入点：Parquet/Iceberg 湖仓布局下“单站长时序点查 (Q1)” vs “整片海面状查询 (Q2)”的读放大问题
对比布局：
  - L1_time:        按天分区 (100文件) + ingest_time 排序
  - L2_station:     按天分区 (100文件) + (station_id, obs_time) 排序
  - L3_uni_hilbert: 按天分区 (100文件) + 传统等步长 Hilbert 排序
  - L4_ours:        冷热分层合并 (前90天每10天合并为9文件 + 近10天按天10文件 = 19文件)
                    + 冷区冻结分位点 Hilbert 桶内按 (station_id, obs_time) 排序
                    + 热区按 ingest_time 排序 + 轻量级 RG 路由表 (Parquet)
依赖环境：numpy, pandas, pyarrow, matplotlib (仅绘图部分需要)
"""

import os
import time
import glob
import shutil
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds

# =========================
# 0. 全局实验参数与随机种子
# =========================
np.random.seed(42)

BASE_DIR = "./exp_ocean_data"
PLOT_DIR = "./exp_plots"

N_DAYS = 100
N_STATIONS = 1000
ROWS_PER_STATION_DAY = 100
TOTAL_ROWS = N_DAYS * N_STATIONS * ROWS_PER_STATION_DAY  # 10,000,000 行
RG_SIZE = 10000  # 每个 Row Group 1 万行 (全表共 1,000 个 RG)

# 对象存储仿真参数 (L_open = 20 ms, 带宽 B = 500 MB/s)
L_OPEN_MS = 20.0
BANDWIDTH_MB_S = 500.0


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


def build_station_network():
    """生成 1,000 个测站：80% 近岸密集 + 20% 远海稀疏"""
    n_nearshore = 800
    n_offshore = 200
    # 近岸密集区 (舟山/东海近岸): 2° x 2°
    near_lat = np.random.uniform(29.5, 31.5, n_nearshore)
    near_lon = np.random.uniform(121.5, 123.5, n_nearshore)
    # 远海稀疏区: 15° x 11.5°
    off_lat = np.random.uniform(20.0, 35.0, n_offshore)
    off_lon = np.random.uniform(123.5, 135.0, n_offshore)

    station_lats = np.concatenate([near_lat, off_lat])
    station_lons = np.concatenate([near_lon, off_lon])
    station_names = np.array([f"ST_{i:04d}" for i in range(N_STATIONS)])
    return station_names, station_lats, station_lons


def build_frozen_cdf_tables(station_lats: np.ndarray, station_lons: np.ndarray):
    """构建并冻结静态空间 CDF 分位点查找表 (257 个分位点，约 4 KB)"""
    lat_quantiles = np.quantile(station_lats, np.linspace(0, 1, 257))
    lon_quantiles = np.quantile(station_lons, np.linspace(0, 1, 257))
    return lat_quantiles, lon_quantiles


def compute_quantile_hilbert(lats, lons, lat_q, lon_q, order=4):
    """O(log M) 查冻结分位点表后编码 Hilbert 桶 (order=4 -> 256 个自适应桶)"""
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
# 2. 数据生成与 4 种物理布局写入
# =========================
def generate_and_write_layouts():
    if os.path.exists(BASE_DIR):
        shutil.rmtree(BASE_DIR)
    os.makedirs(BASE_DIR, exist_ok=True)

    station_names, station_lats, station_lons = build_station_network()
    lat_q, lon_q = build_frozen_cdf_tables(station_lats, station_lons)

    station_uni_hilbert = compute_uniform_hilbert(station_lats, station_lons, order=6)
    station_q_hilbert = compute_quantile_hilbert(station_lats, station_lons, lat_q, lon_q, order=4)

    layout_dirs = {
        "L1_time": os.path.join(BASE_DIR, "L1_time"),
        "L2_station": os.path.join(BASE_DIR, "L2_station"),
        "L3_uni_hilbert": os.path.join(BASE_DIR, "L3_uni_hilbert"),
        "L4_ours": os.path.join(BASE_DIR, "L4_ours"),
    }
    for d in layout_dirs.values():
        os.makedirs(d, exist_ok=True)

    schema = pa.schema([
        ("station_id", pa.string()),
        ("obs_time", pa.timestamp("us")),
        ("event_date", pa.string()),
        ("ingest_time", pa.timestamp("us")),
        ("lat", pa.float64()),
        ("lon", pa.float64()),
        ("sst", pa.float64()),
        ("hilbert_bucket", pa.int64()),
    ])

    start_date = pd.Timestamp("2026-01-01")
    base_st_idx = np.repeat(np.arange(N_STATIONS), ROWS_PER_STATION_DAY)

    write_times = {k: 0.0 for k in layout_dirs}
    routing_records = []
    cold_buffer = []

    print(f"=== 1. 生成 {TOTAL_ROWS:,} 行模拟数据并写入 4 种物理布局 ===")
    t0 = time.perf_counter()

    for day in range(N_DAYS):
        cur_date = start_date + pd.Timedelta(days=day)
        date_str = cur_date.strftime("%Y-%m-%d")

        obs_sec = np.tile(np.linspace(0, 86399, ROWS_PER_STATION_DAY, dtype=np.int64), N_STATIONS)
        obs_sec += np.random.randint(-30, 31, size=len(obs_sec))
        obs_sec = np.clip(obs_sec, 0, 86399)
        ingest_sec = obs_sec + np.random.randint(1, 300, size=len(obs_sec))

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
            "sst": 15.0 + 0.3 * (30.0 - lats) + np.random.normal(0, 0.5, size=len(obs_sec)),
            "hilbert_bucket": station_q_hilbert[base_st_idx],
            "_uni_h": station_uni_hilbert[base_st_idx],
        })

        # L1: 按天分区 + ingest_time 排序
        tw = time.perf_counter()
        df_l1 = df_day.sort_values("ingest_time", kind="mergesort").drop(columns=["_uni_h"])
        pq.write_table(
            pa.Table.from_pandas(df_l1, schema=schema, preserve_index=False),
            os.path.join(layout_dirs["L1_time"], f"day_{day:03d}.parquet"),
            row_group_size=RG_SIZE, compression="snappy",
        )
        write_times["L1_time"] += time.perf_counter() - tw

        # L2: 按天分区 + (station_id, obs_time) 排序
        tw = time.perf_counter()
        df_l2 = df_day.sort_values(["station_id", "obs_time"], kind="mergesort").drop(columns=["_uni_h"])
        pq.write_table(
            pa.Table.from_pandas(df_l2, schema=schema, preserve_index=False),
            os.path.join(layout_dirs["L2_station"], f"day_{day:03d}.parquet"),
            row_group_size=RG_SIZE, compression="snappy",
        )
        write_times["L2_station"] += time.perf_counter() - tw

        # L3: 按天分区 + 传统等步长 Hilbert 排序
        tw = time.perf_counter()
        df_l3 = df_day.sort_values(["_uni_h", "obs_time"], kind="mergesort").drop(columns=["_uni_h"])
        pq.write_table(
            pa.Table.from_pandas(df_l3, schema=schema, preserve_index=False),
            os.path.join(layout_dirs["L3_uni_hilbert"], f"day_{day:03d}.parquet"),
            row_group_size=RG_SIZE, compression="snappy",
        )
        write_times["L3_uni_hilbert"] += time.perf_counter() - tw

        # L4: 冷区 (前90天每10天合并) 按 (hilbert_bucket, station_id, obs_time) 排序；
        #     热区 (近10天按天) 严格按 ingest_time 排序
        tw = time.perf_counter()
        df_l4_part = df_day.drop(columns=["_uni_h"])
        if day < 90:
            cold_buffer.append(df_l4_part)
            if (day + 1) % 10 == 0:
                chunk_id = day // 10
                df_merged = pd.concat(cold_buffer, ignore_index=True)
                df_merged = df_merged.sort_values(["hilbert_bucket", "station_id", "obs_time"], kind="mergesort")
                fname = f"cold_chunk_{chunk_id:02d}.parquet"
                fpath = os.path.join(layout_dirs["L4_ours"], fname)
                pq.write_table(
                    pa.Table.from_pandas(df_merged, schema=schema, preserve_index=False),
                    fpath, row_group_size=RG_SIZE, compression="snappy",
                )
                n_rgs = (len(df_merged) + RG_SIZE - 1) // RG_SIZE
                st_arr = df_merged["station_id"].values
                for rg_id in range(n_rgs):
                    for st in np.unique(st_arr[rg_id * RG_SIZE:min((rg_id + 1) * RG_SIZE, len(df_merged))]):
                        routing_records.append((st, fname, rg_id))
                cold_buffer.clear()
        else:
            df_hot = df_l4_part.sort_values("ingest_time", kind="mergesort")
            fname = f"hot_day_{day:03d}.parquet"
            fpath = os.path.join(layout_dirs["L4_ours"], fname)
            pq.write_table(
                pa.Table.from_pandas(df_hot, schema=schema, preserve_index=False),
                fpath, row_group_size=RG_SIZE, compression="snappy",
            )
            n_rgs = (len(df_hot) + RG_SIZE - 1) // RG_SIZE
            st_arr = df_hot["station_id"].values
            for rg_id in range(n_rgs):
                for st in np.unique(st_arr[rg_id * RG_SIZE:min((rg_id + 1) * RG_SIZE, len(df_hot))]):
                    routing_records.append((st, fname, rg_id))
        write_times["L4_ours"] += time.perf_counter() - tw

    # 写入轻量级 RG 路由表
    tw = time.perf_counter()
    df_route = pd.DataFrame(routing_records, columns=["station_id", "file_name", "rg_id"])
    df_route = df_route.sort_values(["station_id", "file_name", "rg_id"])
    route_path = os.path.join(BASE_DIR, "routing_table.parquet")
    pq.write_table(pa.Table.from_pandas(df_route, preserve_index=False), route_path, compression="snappy")
    write_times["L4_ours"] += time.perf_counter() - tw

    print(f"写入完成，总耗时: {time.perf_counter() - t0:.2f}s\n")
    print("=== 2. 物理存储与写入耗时统计 ===")
    for name, d in layout_dirs.items():
        files = sorted(glob.glob(os.path.join(d, "*.parquet")))
        total_bytes = sum(os.path.getsize(f) for f in files)
        total_rgs = sum(pq.ParquetFile(f).metadata.num_row_groups for f in files)
        print(f"{name:15s} | 写入耗时: {write_times[name]:6.2f}s | 文件数: {len(files):3d} | 总RG数: {total_rgs:4d} | 总大小: {total_bytes/1024/1024:6.2f} MB")
    route_size_kb = os.path.getsize(route_path) / 1024
    print(f"轻量级RG路由表  | 记录数: {len(df_route):7,d} 条 | 文件大小: {route_size_kb:.2f} KB ({route_size_kb/1024:.3f} MB)\n")

    return layout_dirs, route_path, station_lats, station_lons


# =========================
# 3. 剪枝统计、真实查询与对象存储仿真
# =========================
def simulate_minmax_pruning_and_bytes(files, filters, cols_to_read):
    """逐文件、逐 Row Group 检查 Parquet Min-Max 统计，累计读取 RG 数与列压缩字节数"""
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
    return files_hit * L_OPEN_MS + (bytes_read / (BANDWIDTH_MB_S * 1024 * 1024)) * 1000.0 + extra_ms


def run_benchmarks(layout_dirs, route_path):
    target_station = "ST_0100"
    q2_lat_min, q2_lat_max = 30.3, 30.6
    q2_lon_min, q2_lon_max = 122.3, 122.6

    results = {"Q1": [], "Q2_cold_day": [], "Q2_cold_hour": [], "Q2_hot_day": [], "Q2_hot_hour": []}

    print("=== 3. 查询实测与对象存储仿真对比 (L_open=20ms, B=500MB/s) ===")
    print(f"\n[Q1] 单近岸站 ({target_station}) 100天时序")
    q1_cols = ["obs_time", "sst"]
    q1_all_cols = ["station_id", "obs_time", "sst"]

    for name, d in layout_dirs.items():
        files = sorted(glob.glob(os.path.join(d, "*.parquet")))
        f_hit, rgs_r, _, rgs_tot, b_read = simulate_minmax_pruning_and_bytes(
            files, {"station_id": ("==", target_station)}, q1_all_cols
        )
        t1 = time.perf_counter()
        tbl = ds.dataset(d, format="parquet").to_table(columns=q1_cols, filter=(ds.field("station_id") == target_station))
        elapsed_ms = (time.perf_counter() - t1) * 1000.0
        sim_ms = eval_sim_time_ms(f_hit, b_read)
        label = f"{name} (P1)" if name == "L4_ours" else name
        results["Q1"].append((label, b_read / 1024 / 1024, sim_ms, elapsed_ms, rgs_r, f_hit))
        print(f"  {name:15s} (路径一:Dataset下推) | 行数:{tbl.num_rows:6d} | 命中文件:{f_hit:3d}/{len(files):3d} | 读取RG:{rgs_r:4d}/{rgs_tot:4d} | 列压缩:{b_read/1024/1024:6.2f} MB | 本地:{elapsed_ms:6.2f} ms | 对象存储:{sim_ms:7.1f} ms")

        if name == "L4_ours":
            t2 = time.perf_counter()
            rt_tbl = pq.read_table(route_path, filters=[("station_id", "==", target_station)])
            t_route_ms = (time.perf_counter() - t2) * 1000.0
            grouped = rt_tbl.to_pandas().groupby("file_name")["rg_id"].apply(list).to_dict()
            exact_f_hit = len(grouped)
            exact_rgs_r, exact_b_read, parts = 0, 0, []
            for fname, rg_ids in grouped.items():
                pf = pq.ParquetFile(os.path.join(d, fname))
                col_idx_map = {pf.schema_arrow.names[i]: i for i in range(pf.metadata.num_columns)}
                for rg_id in rg_ids:
                    exact_rgs_r += 1
                    rg_meta = pf.metadata.row_group(rg_id)
                    for c in q1_all_cols:
                        exact_b_read += rg_meta.column(col_idx_map[c]).total_compressed_size
                sub_tbl = pf.read_row_groups(rg_ids, columns=q1_all_cols)
                parts.append(sub_tbl.filter(pa.compute.equal(sub_tbl["station_id"], target_station)).select(q1_cols))
            final_tbl = pa.concat_tables(parts)
            exact_elapsed_ms = (time.perf_counter() - t2) * 1000.0
            exact_sim_ms = eval_sim_time_ms(exact_f_hit, exact_b_read, extra_ms=20.0)
            results["Q1"].append(("L4_ours (P2)", exact_b_read / 1024 / 1024, exact_sim_ms, exact_elapsed_ms, exact_rgs_r, exact_f_hit))
            print(f"  {name:15s} (路径二:精确RG路由) | 行数:{final_tbl.num_rows:6d} | 命中文件:{exact_f_hit:3d}/{len(files):3d} | 读取RG:{exact_rgs_r:4d}/{rgs_tot:4d} | 列压缩:{exact_b_read/1024/1024:6.2f} MB | 本地:{exact_elapsed_ms:6.2f} ms (路由:{t_route_ms:.2f}ms) | 对象存储:{exact_sim_ms:7.1f} ms")

    q2_configs = [
        ("Q2_cold", "2026-03-15", pd.Timestamp("2026-03-15 10:00:00"), pd.Timestamp("2026-03-15 11:00:00")),
        ("Q2_hot",  "2026-04-06", pd.Timestamp("2026-04-06 10:00:00"), pd.Timestamp("2026-04-06 11:00:00")),
    ]
    q2_cols = ["station_id", "lat", "lon", "sst"]
    q2_day_cols = ["event_date", "station_id", "lat", "lon", "sst"]
    q2_hr_cols = ["event_date", "obs_time", "station_id", "lat", "lon", "sst"]

    for lbl, t_date, h_s, h_e in q2_configs:
        print(f"\n[{lbl}_day] 单日近岸 0.3°x0.3° 快照 ({t_date})")
        day_stat = {"event_date": ("==", t_date), "lat": ("between", (q2_lat_min, q2_lat_max)), "lon": ("between", (q2_lon_min, q2_lon_max))}
        day_flt = ((ds.field("event_date") == t_date) & (ds.field("lat") >= q2_lat_min) & (ds.field("lat") <= q2_lat_max) &
                   (ds.field("lon") >= q2_lon_min) & (ds.field("lon") <= q2_lon_max))
        for name, d in layout_dirs.items():
            files = sorted(glob.glob(os.path.join(d, "*.parquet")))
            f_hit, rgs_r, rgs_hit_f, _, b_read = simulate_minmax_pruning_and_bytes(files, day_stat, q2_day_cols)
            skip_pct = (1.0 - rgs_r / rgs_hit_f) * 100.0 if rgs_hit_f > 0 else 0.0
            t1 = time.perf_counter()
            tbl = ds.dataset(d, format="parquet").to_table(columns=q2_cols, filter=day_flt)
            elapsed_ms = (time.perf_counter() - t1) * 1000.0
            sim_ms = eval_sim_time_ms(f_hit, b_read)
            results[f"{lbl}_day"].append((name, skip_pct, b_read / 1024 / 1024, rgs_r, rgs_hit_f))
            print(f"  {name:15s} | 行数:{tbl.num_rows:5d} | 命中文件:{f_hit:2d}/{len(files):3d} | 读取RG:{rgs_r:3d}/{rgs_hit_f:3d}RG (跳过:{skip_pct:5.1f}%) | 列压缩:{b_read/1024/1024:5.2f} MB | 本地:{elapsed_ms:6.2f} ms | 对象存储:{sim_ms:6.1f} ms")

        print(f"\n[{lbl}_hour] 1小时近岸 0.3°x0.3° 快照 ({h_s} ~ {h_e})")
        hr_stat = {"event_date": ("==", t_date), "obs_time": ("between", (h_s.to_pydatetime(), h_e.to_pydatetime())),
                   "lat": ("between", (q2_lat_min, q2_lat_max)), "lon": ("between", (q2_lon_min, q2_lon_max))}
        hr_flt = ((ds.field("event_date") == t_date) &
                  (ds.field("obs_time") >= pa.scalar(h_s.to_pydatetime(), type=pa.timestamp("us"))) &
                  (ds.field("obs_time") <= pa.scalar(h_e.to_pydatetime(), type=pa.timestamp("us"))) &
                  (ds.field("lat") >= q2_lat_min) & (ds.field("lat") <= q2_lat_max) &
                  (ds.field("lon") >= q2_lon_min) & (ds.field("lon") <= q2_lon_max))
        for name, d in layout_dirs.items():
            files = sorted(glob.glob(os.path.join(d, "*.parquet")))
            f_hit, rgs_r, rgs_hit_f, _, b_read = simulate_minmax_pruning_and_bytes(files, hr_stat, q2_hr_cols)
            skip_pct = (1.0 - rgs_r / rgs_hit_f) * 100.0 if rgs_hit_f > 0 else 0.0
            t1 = time.perf_counter()
            tbl = ds.dataset(d, format="parquet").to_table(columns=q2_cols, filter=hr_flt)
            elapsed_ms = (time.perf_counter() - t1) * 1000.0
            sim_ms = eval_sim_time_ms(f_hit, b_read)
            results[f"{lbl}_hour"].append((name, skip_pct, b_read / 1024 / 1024, rgs_r, rgs_hit_f))
            print(f"  {name:15s} | 行数:{tbl.num_rows:5d} | 命中文件:{f_hit:2d}/{len(files):3d} | 读取RG:{rgs_r:3d}/{rgs_hit_f:3d}RG (跳过:{skip_pct:5.1f}%) | 列压缩:{b_read/1024/1024:5.2f} MB | 本地:{elapsed_ms:6.2f} ms | 对象存储:{sim_ms:6.1f} ms")

    return results


# =========================
# 4. Matplotlib 实验结果对比绘图
# =========================
def plot_benchmark_charts(results):
    import matplotlib.pyplot as plt
    os.makedirs(PLOT_DIR, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 图 1: Q1 列压缩读取字节数与对象存储仿真耗时对比
    q1_labels = [r[0] for r in results["Q1"]]
    q1_mb = [r[1] for r in results["Q1"]]
    q1_sim = [r[2] for r in results["Q1"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = ["#7f8c8d", "#3498db", "#e67e22", "#9b59b6", "#2ecc71"]
    bars1 = ax1.bar(q1_labels, q1_mb, color=colors)
    ax1.set_title("Q1 (单站100天时序) Parquet 列压缩读取量 (MB)")
    ax1.set_ylabel("读取压缩字节数 (MB)")
    ax1.tick_params(axis="x", rotation=18)
    for b in bars1:
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 2, f"{b.get_height():.2f}", ha="center", fontsize=9)

    bars2 = ax2.bar(q1_labels, q1_sim, color=colors)
    ax2.set_title("Q1 对象存储仿真耗时 (ms, L_open=20ms, B=500MB/s)")
    ax2.set_ylabel("仿真耗时 (ms)")
    ax2.tick_params(axis="x", rotation=18)
    for b in bars2:
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 30, f"{b.get_height():.1f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "fig1_q1_bytes_and_sim_latency.png"), dpi=160)
    plt.close()

    # 图 2: Q2 冷区单日 vs 热区 1 小时文件内 RG 跳过率 (%) 对比
    layouts = [r[0] for r in results["Q2_cold_day"]]
    cold_day_skip = [r[1] for r in results["Q2_cold_day"]]
    hot_hour_skip = [r[1] for r in results["Q2_hot_hour"]]

    x = np.arange(len(layouts))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    b1 = ax.bar(x - w/2, cold_day_skip, w, label="Q2_cold_day (冷区单日 0.3°×0.3°)", color="#2980b9")
    b2 = ax.bar(x + w/2, hot_hour_skip, w, label="Q2_hot_hour (热区1小时 0.3°×0.3°)", color="#27ae60")
    ax.set_xticks(x)
    ax.set_xticklabels(layouts)
    ax.set_ylim(0, 108)
    ax.set_ylabel("命中文件内 Row Group 跳过率 (%)")
    ax.set_title("Q2 近岸小海区查询：冷区空间剪枝 vs 热区时间剪枝")
    ax.legend()
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.5, f"{b.get_height():.1f}%", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "fig2_q2_cold_vs_hot_skip_rate.png"), dpi=160)
    plt.close()
    print(f"图表已保存至 {PLOT_DIR}/")


if __name__ == "__main__":
    dirs, r_path, _, _ = generate_and_write_layouts()
    res = run_benchmarks(dirs, r_path)
    plot_benchmark_charts(res)