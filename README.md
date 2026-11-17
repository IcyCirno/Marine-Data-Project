# Marine Data Project

This course research report studies Parquet layouts for marine observing stations through a reproducible benchmark. It asks why retrieving 100 days of observations from one station can scan more data than retrieving a snapshot of a coastal area.

**Course research report (Chinese)** · [PDF](report/marine_data_course_report_v3.pdf) · [Markdown](report/marine_data_course_report_v3.md)

## Experiment

The benchmark generates 10 million synthetic observations from 1,000 stations over 100 days using a fixed random seed. It compares time ordering, station ordering, uniform Hilbert ordering, and a tiered layout. The tiered layout retains time ordering for recent data. It compacts older data into ten-day files, orders stations by a Hilbert key calculated from frozen coordinate quantiles, and uses a date-aware index to find the relevant Parquet row groups in cold files.

| Measurement | Baseline | Proposed layout |
| --- | ---: | ---: |
| Q1 single-station query, candidate row groups | 1,000 | 109 |
| Q1, estimated compressed column chunks | 124.50 MiB | 13.99 MiB |
| Q1, simulated object-store latency | 2,249.0 ms | 428.0 ms |
| Q2 cold one-hour snapshot, estimated compressed column chunks | 0.16 MiB | 1.20 MiB |

The table uses time ordering as the baseline. The 88.8% reduction for Q1 is based on compressed column chunk sizes reported in Parquet metadata. In the controlled cold-file comparison, a fourth-order quantile Hilbert layout selected 7 of 100 row groups, compared with 25 for fourth-order uniform Hilbert and 6 for sixth-order uniform Hilbert. That comparison covers one date and one coastal bounding box.

The code runs on local Parquet files. The object-store figures come from a latency model, and the report discusses Iceberg integration as a deployment design.

## Reproduce the results

Use a Python environment with roughly 1 GB of free disk space for generated Parquet files. The pinned dependencies match the verified Windows run with Python 3.14.6. Run the following commands from the repository root.

~~~shell
python -m pip install -r requirements.txt
python ocean_storage_experiment_v3.py --out-dir reproduction/data --plot-dir reproduction/plots
python reproduction/verify_v3.py
~~~

Start with an empty output data directory. The verifier checks query results across layouts, matches cold routing entries against the data files, and tests date ranges across cold and hot boundaries.

## Supporting material

- [Benchmark output](reproduction/benchmark.log) and [independent verification output](reproduction/verification.log)
- [Q1 figure](figures/fig1_q1_mib_and_sim_latency.png), [Q2 figure](figures/fig2_q2_cold_vs_hot_skip_rate.png), and [cold-layout ablation figure](figures/fig3_cold_ablation_skip_rate.png)
- [Gemini collaboration record](ai-collaboration/gemini_conversation.md) with the initial report and script referenced by the saved conversation
