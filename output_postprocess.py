#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, gc, gzip, pickle, argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ATTENTION_INDEX = 2  # second-layer unnormalized attention


def load_pickle_gz(path):
    with gzip.open(path, "rb") as fp:
        return pickle.load(fp)


def robust_percentile(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if not np.isfinite(x).all():
        raise ValueError("Attention contains NaN/Inf.")
    return (rankdata(x, method="average") / len(x)).astype(np.float32)


def build_lr_lookup(row_col, lig_rec):
    lookup = defaultdict(list)
    for idx, (i, j) in enumerate(row_col):
        ligand = str(lig_rec[idx][0])
        receptor = str(lig_rec[idx][1])
        lookup[(int(i), int(j))].append((ligand, receptor, idx))
    return lookup


def map_attention_edges_to_lr(edge_index, lr_lookup):
    edge_index = np.asarray(edge_index)
    occurrence = defaultdict(int)
    valid_attention_idx, records = [], []

    for att_idx in range(edge_index.shape[1]):
        i = int(edge_index[0, att_idx])
        j = int(edge_index[1, att_idx])
        key = (i, j)
        k = occurrence[key]
        occurrence[key] += 1

        if key not in lr_lookup or k >= len(lr_lookup[key]):
            continue

        ligand, receptor, input_edge_idx = lr_lookup[key][k]
        valid_attention_idx.append(att_idx)
        records.append({
            "input_edge_idx": int(input_edge_idx),
            "from_id": i,
            "to_id": j,
            "ligand": ligand,
            "receptor": receptor,
            "lr_pair": f"{ligand}-{receptor}",
        })

    return np.asarray(valid_attention_idx, dtype=np.int64), records


def barcode_table(barcode_info):
    rows = []
    for node_id, rec in enumerate(barcode_info):
        rows.append({
            "node_id": node_id,
            "cell_id": str(rec[0]),
            "x": float(rec[1]),
            "y": float(rec[2]),
            "z": float(rec[4]) if len(rec) >= 5 else np.nan,
        })
    return pd.DataFrame(rows)


def attach_coordinates(edge_df, cell_df):
    sender = cell_df.rename(columns={
        "node_id": "from_id", "cell_id": "from_cell",
        "x": "sender_x", "y": "sender_y", "z": "sender_z"
    })
    receiver = cell_df.rename(columns={
        "node_id": "to_id", "cell_id": "to_cell",
        "x": "receiver_x", "y": "receiver_y", "z": "receiver_z"
    })

    edge_df = edge_df.merge(sender, on="from_id", how="left", validate="many_to_one")
    edge_df = edge_df.merge(receiver, on="to_id", how="left", validate="many_to_one")

    edge_df["distance_2d"] = np.sqrt(
        (edge_df["receiver_x"] - edge_df["sender_x"])**2 +
        (edge_df["receiver_y"] - edge_df["sender_y"])**2
    )

    if edge_df["sender_z"].notna().any() and edge_df["receiver_z"].notna().any():
        edge_df["distance_3d"] = np.sqrt(
            edge_df["distance_2d"]**2 +
            (edge_df["receiver_z"] - edge_df["sender_z"])**2
        )
    else:
        edge_df["distance_3d"] = np.nan
    return edge_df


def aggregate_cellpairs(edge_df):
    cellpair = (
        edge_df.groupby(["from_id", "to_id"], as_index=False, observed=True)
        .agg(
            communication_sum=("consensus_score", "sum"),
            communication_mean=("consensus_score", "mean"),
            communication_max=("consensus_score", "max"),
            lr_count=("lr_pair", "nunique"),
            stability_mean=("top5_run_frequency", "mean"),
            consensus_std_mean=("consensus_std", "mean"),
            sender_x=("sender_x", "first"),
            sender_y=("sender_y", "first"),
            sender_z=("sender_z", "first"),
            receiver_x=("receiver_x", "first"),
            receiver_y=("receiver_y", "first"),
            receiver_z=("receiver_z", "first"),
            from_cell=("from_cell", "first"),
            to_cell=("to_cell", "first"),
        )
    )

    idx = edge_df.groupby(["from_id", "to_id"], observed=True)["consensus_score"].idxmax()
    dom = edge_df.loc[idx, [
        "from_id", "to_id", "lr_pair", "ligand", "receptor", "consensus_score"
    ]].rename(columns={
        "lr_pair": "dominant_lr",
        "ligand": "dominant_ligand",
        "receptor": "dominant_receptor",
        "consensus_score": "dominant_lr_score",
    })
    return cellpair.merge(dom, on=["from_id", "to_id"], how="left", validate="one_to_one")


def build_node_activity(cell_df, cellpair_df):
    out_stats = cellpair_df.groupby("from_id", observed=True).agg(
        outgoing_strength=("communication_sum", "sum"),
        outgoing_mean=("communication_mean", "mean"),
        outgoing_max=("communication_max", "max"),
        outgoing_partner_count=("to_id", "nunique"),
    )
    in_stats = cellpair_df.groupby("to_id", observed=True).agg(
        incoming_strength=("communication_sum", "sum"),
        incoming_mean=("communication_mean", "mean"),
        incoming_max=("communication_max", "max"),
        incoming_partner_count=("from_id", "nunique"),
    )

    node = cell_df.set_index("node_id").join(out_stats).join(in_stats).fillna(0).reset_index()
    node["total_strength"] = node["outgoing_strength"] + node["incoming_strength"]
    node["net_strength"] = node["outgoing_strength"] - node["incoming_strength"]
    node["directionality"] = node["net_strength"] / (node["total_strength"] + 1e-12)
    return node


def build_lr_summary(edge_df, high_conf_mask):
    temp = edge_df.copy()
    temp["high_confidence"] = np.asarray(high_conf_mask, dtype=np.int8)
    summary = (
        temp.groupby(["ligand", "receptor", "lr_pair"], as_index=False, observed=True)
        .agg(
            candidate_edge_count=("consensus_score", "size"),
            communication_count=("high_confidence", "sum"),
            total_consensus_score=("consensus_score", "sum"),
            mean_consensus_score=("consensus_score", "mean"),
            median_consensus_score=("consensus_score", "median"),
            mean_top5_frequency=("top5_run_frequency", "mean"),
            unique_sender_count=("from_id", "nunique"),
            unique_receiver_count=("to_id", "nunique"),
        )
    )
    summary["high_conf_fraction"] = (
        summary["communication_count"] /
        summary["candidate_edge_count"].clip(lower=1)
    )
    return summary.sort_values(
        ["communication_count", "total_consensus_score"],
        ascending=[False, False]
    ).reset_index(drop=True)


def parse_known_lr(s):
    return [x.strip() for x in s.split(";") if x.strip()] if s else []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_name", type=str, help='Name of the dataset', required=True)
    p.add_argument("--model_name", type=str, help='Provide a model name', required=True)
    p.add_argument('--total_runs', type=int, default=10, help='How many runs for ensemble (at least 2 are preferred)', required=True)
    p.add_argument("--embedding_path", default="embedding_data/")
    p.add_argument("--metadata_from", default="outp/metadata/")
    p.add_argument("--data_from", default="outp/input_graph/")
    p.add_argument("--output_path", default="outputV/")
    p.add_argument("--high_conf_percentile", type=float, equired=True)
    p.add_argument("--min_top5_frequency", type=float, required=True)
    p.add_argument("--known_lr", default="")
    args = p.parse_args()

    metadata_dir = os.path.join(args.metadata_from, args.data_name)
    graph_dir = os.path.join(args.data_from, args.data_name)
    embedding_dir = os.path.join(args.embedding_path, args.data_name)
    output_dir = os.path.join(args.output_path, args.data_name)
    os.makedirs(output_dir, exist_ok=True)

    barcode_info = load_pickle_gz(os.path.join(metadata_dir, args.data_name + "_barcode_info"))
    row_col, edge_weight, lig_rec, total_num_cell, w1, w2, w3 = load_pickle_gz(
        os.path.join(graph_dir, args.data_name + "_adjacency_records2")
    )

    cell_df = barcode_table(barcode_info)
    lr_lookup = build_lr_lookup(row_col, lig_rec)

    run1 = load_pickle_gz(os.path.join(embedding_dir, args.model_name + "_r1_attention"))
    valid_idx, base_records = map_attention_edges_to_lr(run1[0], lr_lookup)
    base_df = pd.DataFrame(base_records)
    n_edges = len(base_df)
    if n_edges == 0:
        raise RuntimeError("No LR edges matched.")

    raw_runs = np.full((n_edges, args.total_runs), np.nan, dtype=np.float32)
    pct_runs = np.full((n_edges, args.total_runs), np.nan, dtype=np.float32)
    ref_edge_index = np.asarray(run1[0])[:, valid_idx]

    for r in range(1, args.total_runs + 1):
        bundle = load_pickle_gz(
            os.path.join(embedding_dir, args.model_name + f"_r{r}_attention")
        )
        edge_index_r = np.asarray(bundle[0])[:, valid_idx]
        if not np.array_equal(edge_index_r, ref_edge_index):
            raise ValueError(f"Run {r}: edge order differs from run 1.")

        raw_all = np.asarray(bundle[ATTENTION_INDEX]).reshape(-1)
        raw = raw_all[valid_idx].astype(np.float64)

        raw_runs[:, r - 1] = raw
        pct_runs[:, r - 1] = robust_percentile(raw)

        del bundle
        gc.collect()

    edge_df = base_df.copy()
    edge_df["attention_raw_mean"] = np.nanmean(raw_runs, axis=1)
    edge_df["attention_raw_median"] = np.nanmedian(raw_runs, axis=1)
    edge_df["attention_raw_std"] = np.nanstd(raw_runs, axis=1, ddof=1)
    edge_df["consensus_score"] = np.nanmedian(pct_runs, axis=1)
    edge_df["consensus_mean"] = np.nanmean(pct_runs, axis=1)
    edge_df["consensus_std"] = np.nanstd(pct_runs, axis=1, ddof=1)
    edge_df["top1_run_frequency"] = np.nanmean(pct_runs >= 0.99, axis=1)
    edge_df["top5_run_frequency"] = np.nanmean(pct_runs >= 0.95, axis=1)
    edge_df["top10_run_frequency"] = np.nanmean(pct_runs >= 0.90, axis=1)

    for r in range(args.total_runs):
        edge_df[f"attention_raw_r{r+1}"] = raw_runs[:, r]
        edge_df[f"attention_pct_r{r+1}"] = pct_runs[:, r]

    edge_df["consensus_rank"] = edge_df["consensus_score"].rank(
        method="average", ascending=False
    )

    edge_df = attach_coordinates(edge_df, cell_df)

    high_conf_mask = (
        (edge_df["consensus_score"] >= args.high_conf_percentile) &
        (edge_df["top5_run_frequency"] >= args.min_top5_frequency)
    )
    edge_df["high_confidence"] = high_conf_mask.astype(np.int8)

    cellpair_df = aggregate_cellpairs(edge_df)
    node_df = build_node_activity(cell_df, cellpair_df)
    lr_summary_df = build_lr_summary(edge_df, high_conf_mask)

    known_lr = parse_known_lr(args.known_lr)
    known_lr_edges = edge_df[edge_df["lr_pair"].isin(known_lr)].copy() if known_lr else edge_df.iloc[0:0].copy()
    known_lr_summary = lr_summary_df[lr_summary_df["lr_pair"].isin(known_lr)].copy() if known_lr else lr_summary_df.iloc[0:0].copy()

    prefix = os.path.join(output_dir, args.model_name)

    try:
        edge_df.to_parquet(prefix + "_01_all_lr_edges.parquet", index=False)
        cellpair_df.to_parquet(prefix + "_02_cellpair_communication.parquet", index=False)
    except Exception:
        edge_df.to_csv(prefix + "_01_all_lr_edges.csv.gz", index=False, compression="gzip")
        cellpair_df.to_csv(prefix + "_02_cellpair_communication.csv.gz", index=False, compression="gzip")

    edge_df.loc[high_conf_mask].to_csv(prefix + "_03_high_confidence_lr_edges.csv", index=False)
    node_df.to_csv(prefix + "_04_cell_communication_activity.csv", index=False)
    lr_summary_df.to_csv(prefix + "_05_lr_summary.csv", index=False)
    lr_summary_df.head(50).to_csv(prefix + "_08_top50_lr_for_barplot.csv", index=False)

    if known_lr:
        known_lr_edges.to_csv(prefix + "_06_known_lr_edges.csv", index=False)
        known_lr_summary.to_csv(prefix + "_07_known_lr_summary.csv", index=False)

    print("Done")
    print("All LR edges:", len(edge_df))
    print("High-confidence LR edges:", int(high_conf_mask.sum()))
    print("Cell-pair edges:", len(cellpair_df))
    print("Cells:", len(node_df))
    print("LR pairs:", len(lr_summary_df))
    print("Output:", output_dir)


if __name__ == "__main__":
    main()
