"""Validate the v3 benchmark outputs in reproduction_20261116/data."""

from collections import Counter
from datetime import datetime, timedelta
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import patch
import json

import pyarrow.dataset as ds
import pyarrow.parquet as pq


source = Path(__file__).resolve().parent.parent / "ocean_storage_experiment_v3.py"
spec = spec_from_file_location("ocean_experiment_v3", source)
exp = module_from_spec(spec)
spec.loader.exec_module(exp)

root = Path(__file__).resolve().parent / "data"
layouts = {name: root / name for name in ("L1_time", "L2_station", "L3_uni_hilbert", "L4_ours")}
l4 = layouts["L4_ours"]
route_path = root / "cold_routing_table.parquet"
read_cols = ["obs_time", "sst"]
station = "ST_0100"
q1_datasets = {name: ds.dataset(path, format="parquet") for name, path in layouts.items()}


def canonical(table):
    keys = [(name, "ascending") for name in table.column_names]
    return table.sort_by(keys).combine_chunks()


def q1_expr(station_id, date_start, date_end):
    return (
        (ds.field("station_id") == station_id)
        & (ds.field("event_date") >= date_start)
        & (ds.field("event_date") <= date_end)
    )


expr = q1_expr(station, "2026-01-01", "2026-04-10")
full_p1 = [canonical(dataset.to_table(columns=read_cols, filter=expr))
           for dataset in q1_datasets.values()]
full_p2 = canonical(exp.query_l4_exact_q1(str(l4), str(route_path), station, "2026-01-01", "2026-04-10"))
assert full_p2.num_rows == 10_000 and all(t.equals(full_p2) for t in full_p1)
print(f"Q1 full range: 5 paths return the same {full_p2.num_rows:,} rows and values")

for name, path in layouts.items():
    files = sorted(path.glob("*.parquet"))
    assert all(pq.read_schema(p).remove_metadata().equals(exp.CORE_SCHEMA) for p in files)
    assert sum(pq.ParquetFile(p).metadata.num_rows for p in files) == 10_000_000, name
print("All 4 layouts have 10,000,000 rows and matching 7-column schema")

real_parquet_file = pq.ParquetFile
cold_reads = []


class TrackingParquetFile(real_parquet_file):
    def __init__(self, source, *args, **kwargs):
        super().__init__(source, *args, **kwargs)
        self.file_name = Path(source).name

    def read_row_groups(self, row_groups, *args, **kwargs):
        if self.file_name.startswith("cold_"):
            cold_reads.extend((self.file_name, rg) for rg in row_groups)
        return super().read_row_groups(row_groups, *args, **kwargs)


cases = (
    ("hot-only", station, "2026-04-06", "2026-04-06", 100, 0),
    ("one cold day", station, "2026-03-15", "2026-03-15", 100, 1),
    ("two cold chunks", station, "2026-01-10", "2026-01-11", 200, 2),
    ("cold/hot boundary", station, "2026-03-31", "2026-04-01", 200, 1),
    ("no matching dates", station, "2026-05-01", "2026-05-01", 0, 0),
    ("missing station", "ST_9999", "2026-01-01", "2026-04-10", 0, 0),
)
for label, station_id, start, end, expected_count, expected_rg in cases:
    pred = q1_expr(station_id, start, end)
    expected = canonical(q1_datasets["L1_time"].to_table(columns=read_cols, filter=pred))
    cold_reads.clear()
    with (
        patch.object(pq, "ParquetFile", TrackingParquetFile),
        patch.object(pq, "read_table", wraps=pq.read_table) as read_table,
    ):
        actual, files_reported, rgs_reported = exp.query_l4_exact_q1(
            str(l4), str(route_path), station_id, start, end, return_stats=True)
    actual = canonical(actual)
    assert actual.num_rows == expected_count and actual.equals(expected), label
    files_seen = len({name for name, _ in cold_reads})
    assert files_seen == len(cold_reads) == expected_rg == rgs_reported == files_reported, label
    route_reads = sum(Path(c.args[0]) == route_path for c in read_table.call_args_list)
    print(f"Q1 {label}: identical {actual.num_rows} rows, {len(cold_reads)} cold RGs read in "
          f"{files_seen} cold data files; {route_reads} cold route table lookups")

routing_records = pq.read_table(route_path).to_pylist()
actual_route = {(r["station_id"], r["file_name"], r["rg_id"]) for r in routing_records}
assert len(actual_route) == len(routing_records) == 9000
expected_route = set()
file_ranges = {}
hot_station_counts = Counter()
for path in sorted(l4.glob("*.parquet")):
    with pq.ParquetFile(path) as file:
        if path.name.startswith("cold_"):
            date_index = file.schema_arrow.get_field_index("event_date")
            dates = [file.metadata.row_group(i).column(date_index).statistics
                     for i in range(file.num_row_groups)]
            assert all(stat.has_min_max for stat in dates)
            file_ranges[path.name] = (min(stat.min for stat in dates), max(stat.max for stat in dates))
        for index in range(file.num_row_groups):
            labels = set(file.read_row_group(index, columns=["station_id"])["station_id"].to_pylist())
            if path.name.startswith("cold_"):
                expected_route.update((label, path.name, index) for label in labels)
            else:
                hot_station_counts[len(labels)] += 1
assert actual_route == expected_route
assert all((r["min_date"], r["max_date"]) == file_ranges[r["file_name"]] for r in routing_records)
print("Cold route intervals: all 9,000 entries match their files' actual date bounds")
assert hot_station_counts == {1000: 100}
print(f"Cold route: 9,000 unique and correct mappings, {route_path.stat().st_size:,} bytes")
print("Hot layout: every one of the 100 hot RGs contains all 1,000 stations")
cdf = root / "frozen_cdf_table.json"
quantiles = json.loads(cdf.read_text(encoding="utf-8"))
assert all(len(quantiles[k]) == 257 for k in ["lat_quantiles", "lon_quantiles"])
print(f"CDF: 257 lat and 257 lon boundaries, {cdf.stat().st_size:,} bytes")

geo = (
    (ds.field("lat") >= 30.3) & (ds.field("lat") <= 30.6)
    & (ds.field("lon") >= 122.3) & (ds.field("lon") <= 122.6)
)
q2_cols = ["station_id", "lat", "lon", "sst"]
for label, date in (("cold", "2026-03-15"), ("hot", "2026-04-06")):
    for window in ("day", "hour"):
        pred = geo & (ds.field("event_date") == date)
        if window == "hour":
            low = datetime.fromisoformat(f"{date} 10:00:00")
            high = datetime.fromisoformat(f"{date} 11:00:00")
            pred = pred & (ds.field("obs_time") >= low) & (ds.field("obs_time") <= high)
        tables = [canonical(d.to_table(columns=q2_cols, filter=pred)) for d in q1_datasets.values()]
        assert all(t.equals(tables[0]) for t in tables), (label, window)
        print(f"Q2 {label} {window}: four layouts return identical {tables[0].num_rows:,} rows")
        if label == "cold" and window == "day":
            for path in sorted((root / "ablation_cold_chunk07").glob("*.parquet")):
                t = ds.dataset(path, format="parquet").to_table(columns=q2_cols, filter=pred)
                assert canonical(t).equals(tables[0])
            print("Q2 cold day: three ablation variants return the same rows")

hour_counts = Counter()
hot_file = l4 / "hot_day_095.parquet"
hot_day = datetime(2026, 4, 6)
for hour in range(24):
    low = hot_day + timedelta(hours=hour)
    high = low + timedelta(hours=1)
    filt = {"event_date": ("==", "2026-04-06"), "obs_time": ("between", (low, high)),
            "lat": ("between", (30.3, 30.6)), "lon": ("between", (122.3, 122.6))}
    _, groups, _, _, _ = exp.estimate_footer_pruning_bytes([hot_file], filt, ["event_date", "obs_time", *q2_cols])
    hour_counts[groups] += 1
print(f"Hot day 04-06, candidates over 24 hourly windows: {dict(sorted(hour_counts.items()))}")

print("Compressed column chunk totals (MiB) across all files")
storage = {}
for name in ("L1_time", "L4_ours"):
    column_bytes = Counter()
    for path in sorted(layouts[name].glob("*.parquet")):
        with pq.ParquetFile(path) as file:
            for index in range(file.num_row_groups):
                group = file.metadata.row_group(index)
                for col in range(group.num_columns):
                    meta = group.column(col)
                    column_bytes[meta.path_in_schema] += meta.total_compressed_size
    storage[name] = column_bytes
    print(f"  {name}: " + ", ".join(f"{k}={v / exp.MIB:.4f}" for k, v in column_bytes.items()))
l1, l4_bytes = storage["L1_time"], storage["L4_ours"]
diff = sum(l4_bytes.values()) - sum(l1.values())
t_delta = sum(l4_bytes[k] - l1[k] for k in ("obs_time", "ingest_time"))
print(f"  Total increase={diff / exp.MIB:.4f} MiB; timestamp columns={t_delta / exp.MIB:.4f} MiB "
      f"({t_delta / diff * 100:.1f}% of column chunk increase)")
