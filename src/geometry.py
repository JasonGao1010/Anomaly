"""Measure observable geometry and normal-only conditional probes on real STU scans."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack
from concurrent.futures import ProcessPoolExecutor
import csv
import ctypes
import gc
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import tempfile
import time
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np

from .data import _atomic_json, host_disk
from .evaluate import evaluation_targets
from .profile import GEOMETRY_PARAMETERS, observed_geometry
from .probes import FEATURES, GEOMETRY, fit_reference, score_reference
from .protocol import load_protocol
from .scene import STUSequence, LabelMode


SEED, NORMAL_QUERIES = 20260911, 8192
FLAGS = ("valid", "condition_valid", "normal_valid", "normal_change_valid")
DTYPE = np.dtype([("source_slot", "i4"), ("frame", "i4"), ("target", "i1"),
                  ("semantic", "u2"), *[(k, "f4") for k in FEATURES],
                  *[(k, "?") for k in FLAGS]])
READER = {}
REFERENCE, THRESHOLDS = None, None
GROUP_DTYPE = np.dtype([("bits", "u4"), ("count", "i8"), ("positive", "i8")])
SAMPLE_DTYPE = np.dtype(DTYPE.descr + [("sequence", "i4"), ("weight", "f8")])


def reader(data_root, partition, sequence):
    key = (str(data_root), partition, sequence)
    if READER.get("key") != key:
        READER.clear()
        READER.update(key=key, source=STUSequence.open(
            data_root, protocol=load_protocol(), partition=partition,
            sequence_id=sequence, label_mode=LabelMode.REQUIRED))
    return READER["source"]


def extract_frame(job):
    data_root, output, partition, sequence, frame_id = job
    started = time.monotonic()
    source = reader(data_root, partition, sequence)[frame_id]
    semantic = source.labels.semantic
    target = evaluation_targets(source.xyzi[:, :3], semantic)
    eligible = int(np.count_nonzero(target == 1)) >= 5
    row = dict(partition=partition, sequence=sequence, frame=frame_id,
               actual_returns=len(source.real_slots), eligible=eligible,
               official_normal=int(np.count_nonzero(target == 0)),
               official_anomaly=int(np.count_nonzero(target == 1)), rows=0)
    if partition == "train" or eligible:
        slots = source.real_slots
        distance = np.linalg.norm(source.xyzi[slots, :3], axis=1)
        query = slots[(distance >= 2.5) & (distance <= 50)]
        if partition == "train" and len(query) > NORMAL_QUERIES:
            # Sampling precedes label filtering; every scan uses its own random stream.
            rng = np.random.default_rng(np.random.SeedSequence([SEED, sequence, frame_id]))
            query = np.sort(rng.choice(query, NORMAL_QUERIES, replace=False))
        geometry = observed_geometry(source.xyzi[slots], slots, query_slots=query)
        if partition == "train":
            mapping = load_protocol().semantic_class_map
            keep = np.array([mapping.get(int(x), 255) != 255 for x in semantic[query]])
            labels = np.zeros(int(keep.sum()), np.int8)
        else:
            keep = target[query] >= 0
            labels = target[query][keep].astype(np.int8)
        values = np.empty(int(keep.sum()), DTYPE)
        for name in (*FEATURES, *FLAGS, "source_slot"):
            values[name] = geometry[name][keep]
        values["frame"], values["target"] = frame_id, labels
        values["semantic"] = semantic[query][keep]
        path = Path(output) / "features" / partition / str(sequence) / f"{frame_id:06d}.npy"
        np.save(path, values, allow_pickle=False)
        row.update(rows=len(values), geometry_valid=int(values["valid"].sum()),
                   all_geometry_valid=int(np.isfinite(np.column_stack([values[k] for k in GEOMETRY])).all(axis=1).sum()),
                   bytes=path.stat().st_size)
        if partition == "val" and (int((labels == 1).sum()) != row["official_anomaly"]
                                    or int((labels == 0).sum()) != row["official_normal"]):
            raise ValueError("observable point identities do not match the official point set")
    row.update(seconds=time.monotonic()-started,
               max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    return row


def extract(data_root, output, workers):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "features"
    if destination.exists():
        raise FileExistsError("feature extraction already exists; use the analysis stage to reuse it")
    protocol = load_protocol()
    sequences = [("train", 206), ("train", 201), *[("val", s) for s in protocol.public_sequence_ids]]
    jobs = []
    for partition, sequence in sequences:
        source = reader(data_root, partition, sequence)
        (destination / partition / str(sequence)).mkdir(parents=True)
        jobs.extend((str(data_root), str(output), partition, sequence, f) for f in source.frame_ids)
    READER.clear()
    disk = host_disk()
    # Persistent features plus exact score-count aggregates and report peaks.
    peak = 26_000_000_000
    if disk["SizeRemaining"] - peak < disk["reserve_bytes"]:
        raise OSError("the 26 GB peak analysis budget would invade the physical E: reserve")
    config = dict(format="stu-observed-geometry", data_root=str(Path(data_root).resolve()),
                  parameters=GEOMETRY_PARAMETERS, seed=SEED, normal_queries_per_frame=NORMAL_QUERIES,
                  normal_selection="uniform current-scan range-valid slots, before semantic filtering",
                  normal_labels="valid normal_semantic_class_map classes only",
                  val_scope="every official eligible frame; every official valid point; neighborhoods use full actual scan",
                  normal_fit="train/206", normal_transfer="train/201", evaluation="public development val19",
                  workers=workers, cpu_affinity=len(os.sched_getaffinity(0)), library_threads=1,
                  peak_write_budget_bytes=peak, initial_disk=disk,
                  implementation_pilot=dict(frames=[0,224,448], partition="train", sequence=206,
                                            label_mode="forbidden", workers_1_vs_4="all returned arrays exactly equal",
                                            single_thread_seconds=[1.0193,1.0448,.9602],
                                            four_thread_seconds=[.7588,.7548,.7421]))
    _atomic_json(output / "configuration.json", config)
    start, records, minimum_disk = time.monotonic(), [], disk["SizeRemaining"]
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        for row in pool.map(extract_frame, jobs, chunksize=4):
            records.append(row)
            if len(records) % 100 == 0 or len(records) == len(jobs):
                volume = host_disk()
                minimum_disk = min(minimum_disk, volume["SizeRemaining"])
                print(json.dumps(dict(stage="geometry", completed=len(records), total=len(jobs),
                                      seconds=round(time.monotonic()-start,2),
                                      disk_remaining=volume["SizeRemaining"])), flush=True)
    _atomic_json(output / "extraction.json", dict(
        frames=records, seconds=time.monotonic()-start, minimum_disk_remaining=minimum_disk,
        final_disk=host_disk(), bytes=sum(r.get("bytes",0) for r in records)))


def as_data(values):
    return {key: values[key] for key in values.dtype.names}


def score_groups(scores, labels):
    """Compress identical float32 scores exactly; never round into histogram bins."""
    scores = np.asarray(scores, np.float32)
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("score groups require finite empirical scores")
    bits, inverse, count = np.unique(scores.view(np.uint32), return_inverse=True, return_counts=True)
    result = np.empty(len(bits), GROUP_DTYPE)
    result["bits"], result["count"] = bits, count
    result["positive"] = np.bincount(inverse[labels == 1], minlength=len(bits))
    return result


def merge_groups(parts):
    if not parts:
        return np.empty(0, GROUP_DTYPE)
    values = np.concatenate(parts)
    if not len(values):
        return values
    values = values[np.argsort(values["bits"],kind="stable")]
    starts = np.r_[0, np.flatnonzero(values["bits"][1:] != values["bits"][:-1])+1]
    result = np.empty(len(starts), GROUP_DTYPE)
    result["bits"] = values["bits"][starts]
    for name in ("count", "positive"):
        result[name] = np.add.reduceat(values[name], starts)
    return result


def subtract_groups(total, removed):
    result = total.copy()
    indices = np.searchsorted(result["bits"], removed["bits"])
    if not np.array_equal(result["bits"][indices], removed["bits"]):
        raise ValueError("sequence scores are absent from pooled score identities")
    for name in ("count", "positive"):
        result[name][indices] -= removed[name]
    if np.any(result["positive"] < 0) or np.any(result["count"] < result["positive"]):
        raise ValueError("negative sequence-subtraction count")
    return result[result["count"] > 0]


def group_metrics(groups):
    from .evaluate import metrics_from_groups
    positive = int(groups["positive"].sum())
    negative = int(groups["count"].sum()) - positive
    descending = groups[::-1]
    iterator = ((a["bits"], a["count"], a["positive"])
                for start in range(0,len(descending),1<<18)
                if len(a := descending[start:start+(1<<18)]))
    return metrics_from_groups(iterator, positive=positive, negative=negative)


def normal_threshold(groups):
    descending = groups[::-1]
    counts = np.cumsum(descending["count"])
    feasible = np.flatnonzero(counts <= .01 * int(groups["count"].sum()))
    return float(descending["bits"][feasible[-1]].view(np.float32)) if len(feasible) else None


def threshold_counts(groups, threshold):
    take = np.zeros(len(groups), bool) if threshold is None else groups["bits"].view(np.float32) >= threshold
    positive, total = int(groups["positive"].sum()), int(groups["count"].sum())
    tp, detected = int(groups["positive"][take].sum()), int(groups["count"][take].sum())
    return dict(normal=total-positive, anomaly=positive, tp=tp, fp=detected-tp,
                FPR=100*(detected-tp)/(total-positive) if total>positive else None,
                recall=100*tp/positive if positive else None)


def comparisons(scores, individual):
    for mode, control in (("range","A_range"),("direction","A_direct"),("sampling","A_sampling")):
        names = (control,"B_geometry","C_"+mode)
        finite = np.logical_and.reduce([np.isfinite(scores[k]) for k in names])
        yield mode, {k:scores[k] for k in names}, finite
    finite = np.isfinite(scores["B_geometry"]) & np.isfinite(scores["B_normalized"])
    yield "normalization", {k:scores[k] for k in ("B_geometry","B_normalized")}, finite
    for feature in GEOMETRY:
        for mode in ("range","direction","sampling"):
            names = ("B_geometry","C_"+mode)
            fields = {k:individual[k][feature] for k in names}
            finite = np.logical_and.reduce([np.isfinite(a) for a in fields.values()])
            yield f"feature/{feature}/{mode}", fields, finite


def merge_count_archives(chunks,destination,keys):
    """Stream one exact comparison at a time; inputs are bounded frame blocks."""
    with ExitStack() as stack:
        archives=[stack.enter_context(np.load(p,allow_pickle=False)) for p in chunks]
        archive=stack.enter_context(ZipFile(destination,"w",ZIP_DEFLATED,allowZip64=True))
        for key_index,key in enumerate(keys):
            total=np.empty(0,GROUP_DTYPE)
            for part in archives:
                total=merge_groups([total,part[str(key_index)]])
            with archive.open(str(key_index)+".npy","w",force_zip64=True) as member:
                np.lib.format.write_array(member,total,allow_pickle=False)
            del total
            ctypes.CDLL(None).malloc_trim(0)


def process_scores(job):
    output, partition, sequence = job
    output = Path(output)
    directory = output / "counts" / partition / str(sequence)
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/"summary.json").exists():
        if not (directory/"groups.npz").exists():
            raise ValueError("completed scoring is missing its exact counts")
        return dict(partition=partition,sequence=sequence,seconds=0.,max_rss_bytes=0,reused_complete=True)
    pending, coverage, confusion = defaultdict(list), defaultdict(lambda:np.zeros(4,np.int64)), defaultdict(lambda:np.zeros(3,np.int64))
    workspace=tempfile.TemporaryDirectory(prefix="counts_",dir=directory)
    chunks=[]
    candidates, samples, frames = [], [], []
    started = time.monotonic()
    paths = sorted((output/"features"/partition/str(sequence)).glob("*.npy"))
    for index, path in enumerate(paths):
        values = np.load(path,allow_pickle=False)
        scores, individual, cells = score_reference(as_data(values),REFERENCE)
        labels = values["target"]
        anomaly_count = int((labels==1).sum())
        strata = {"all":np.ones(len(values),bool)}
        range_bins = np.searchsorted([10,20,35],values["range"],side="right")
        for k in range(4):
            strata["range_"+str(k)] = range_bins==k
        if partition=="val":
            strata["returns_"+str(np.searchsorted([20,100,500],anomaly_count,side="right"))] = strata["all"]
        for cohort, methods, finite in comparisons(scores,individual):
            for group, mask in strata.items() if not cohort.startswith("feature/") else [("all",strata["all"])]:
                key = cohort+"|"+group
                coverage[key] += [int(np.sum(mask & (labels==0))),int(np.sum(mask & (labels==1))),
                                  int(np.sum(mask & finite & (labels==0))),int(np.sum(mask & finite & (labels==1)))]
            for method, score in methods.items():
                key = cohort+"|"+method
                pending[key].append(score_groups(score[finite],labels[finite]))
                threshold = None if THRESHOLDS is None else THRESHOLDS.get(key)
                detected = finite & (score>=threshold) if threshold is not None else np.zeros(len(values),bool)
                if not cohort.startswith("feature/"):
                    for semantic in np.unique(values["semantic"][finite & (labels==0)]):
                        subset = (values["semantic"]==semantic) & (labels==0) & finite
                        confusion[key+"|"+str(int(semantic))] += [int(subset.sum()),int((subset & detected).sum()),0]
                    if partition=="val":
                        frames.append(dict(sequence=sequence,frame=int(values["frame"][0]),cohort=cohort,method=method,
                                           normal=int(np.sum(finite & (labels==0))),anomaly=int(np.sum(finite & (labels==1))),
                                           fp=int(np.sum(detected & (labels==0))),tp=int(np.sum(detected & (labels==1)))))
                        if method in ("C_direction","C_sampling"):
                            normal=np.flatnonzero(detected & (labels==0))
                            if len(normal):
                                top=normal[np.argmax(score[normal])]
                                candidates.append(dict(sequence=sequence,frame=int(values["frame"][top]),cohort=cohort,
                                                       method=method,fp=len(normal),slot=int(values["source_slot"][top]),
                                                       score=float(score[top])))
        if partition=="val":
            normal=np.flatnonzero(labels==0)
            rng=np.random.default_rng(np.random.SeedSequence([SEED,sequence,int(values["frame"][0]),91]))
            take=np.sort(rng.choice(normal,min(2048,len(normal)),replace=False))
            selected=np.r_[np.flatnonzero(labels==1),take]
            sample=np.empty(len(selected),SAMPLE_DTYPE)
            for name in DTYPE.names:
                sample[name]=values[name][selected]
            sample["sequence"],sample["weight"]=sequence,1.
            sample["weight"][sample["target"]==0]=len(normal)/len(take) if len(take) else 0
            samples.append(sample)
        # Fixed-size blocks prevent 47 growing sequence distributions occupying RAM together.
        if (index+1)%8==0 or index+1==len(paths):
            keys=sorted(pending)
            chunk=Path(workspace.name)/f"{len(chunks):04d}.npz"
            with ZipFile(chunk,"w",ZIP_DEFLATED,allowZip64=True) as archive:
                for key_index,key in enumerate(keys):
                    group=merge_groups(pending.pop(key))
                    with archive.open(str(key_index)+".npy","w",force_zip64=True) as member:
                        np.lib.format.write_array(member,group,allow_pickle=False)
                del group
            chunks.append(chunk)
            pending.clear()
            ctypes.CDLL(None).malloc_trim(0)
        if (index+1)%25==0 or index+1==len(paths):
            print(json.dumps(dict(stage="score_frames",partition=partition,sequence=sequence,
                                  completed=index+1,total=len(paths),seconds=round(time.monotonic()-started,2))),flush=True)
    merge_count_archives(chunks,directory/"groups.npz",keys)
    workspace.cleanup()
    if samples:
        np.save(output/"samples"/f"{sequence}.npy",np.concatenate(samples),allow_pickle=False)
    totals=dict(partition=partition,sequence=sequence,keys=keys,frames=frames,
                coverage={k:v.tolist() for k,v in coverage.items()},
                confusion={k:v.tolist() for k,v in confusion.items()},candidates=candidates,
                seconds=time.monotonic()-started,
                max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    _atomic_json(directory/"summary.json",totals)
    return dict(partition=partition,sequence=sequence,seconds=totals["seconds"],max_rss_bytes=totals["max_rss_bytes"])


def load_counts(output,partition,sequence):
    directory=Path(output)/"counts"/partition/str(sequence)
    summary=json.loads((directory/"summary.json").read_text())
    with np.load(directory/"groups.npz") as data:
        groups={k:data[str(i)] for i,k in enumerate(summary["keys"])}
    return groups,summary


def csv_rows(path,rows):
    if not rows:
        return
    with Path(path).open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(output,workers=6):
    global REFERENCE,THRESHOLDS
    output=Path(output)
    config=json.loads((output/"configuration.json").read_text())
    extraction=json.loads((output/"extraction.json").read_text())
    disk=host_disk()
    if disk["SizeRemaining"]-14_000_000_000<disk["reserve_bytes"]:
        raise OSError("score counts and temporary blocks would invade the physical E: reserve")
    if config["parameters"]!=GEOMETRY_PARAMETERS or config["normal_fit"]!="train/206":
        raise ValueError("feature definition or fitting source changed")
    start=time.monotonic()
    normal=np.concatenate([np.load(p,allow_pickle=False) for p in sorted((output/"features/train/206").glob("*.npy"))])
    if np.any(normal["target"]!=0) or len(np.unique(normal["frame"]))!=449:
        raise ValueError("reference requires all 449 train/206 frames and only valid normal labels")
    REFERENCE=fit_reference(as_data(normal))
    existing=output/"reference.json"
    if existing.exists() and json.loads(existing.read_text())!=REFERENCE["metadata"]:
        raise ValueError("normal reference changed; cached scoring cannot be reused")
    _atomic_json(existing,REFERENCE["metadata"])
    del normal
    THRESHOLDS=None
    process_scores((str(output),"train",206))
    fit_groups,_=load_counts(output,"train",206)
    THRESHOLDS={key:normal_threshold(group) for key,group in fit_groups.items()}
    _atomic_json(output/"thresholds.json",dict(source="train/206 fitted normal scores",normal_fpr_limit=.01,values=THRESHOLDS))
    del fit_groups
    gc.collect()
    # Release allocator-retained fit/count workspaces before creating reader workers.
    ctypes.CDLL(None).malloc_trim(0)
    (output/"samples").mkdir(exist_ok=True)
    jobs=[(str(output),"train",201),*[(str(output),"val",s) for s in load_protocol().public_sequence_ids]]
    # Fork shares immutable normal reference arrays; each worker holds one scan.
    resources=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context("fork")) as pool:
        for row in pool.map(process_scores,jobs,chunksize=1):
            resources.append(row)
            disk=host_disk()
            print(json.dumps(dict(stage="scores",**row,disk_remaining=disk["SizeRemaining"])),flush=True)
    from .profile_report import sampling_increment, summarize_geometry
    summarize_geometry(output,extraction)
    increment=sampling_increment(output,REFERENCE)
    from .probes import descriptive_matches
    sample=np.concatenate([np.load(p,allow_pickle=False) for p in sorted((output/"samples").glob("*.npy"))])
    matched=descriptive_matches(as_data(sample),REFERENCE)
    _atomic_json(output/"matched.json",dict(
        scope="descriptive weighted sample of official eligible val frames; not official metrics",
        normal_sampling="up to 2048 uniform normal slots per frame with inverse sampling probability weights",
        anomaly_sampling="all official anomaly points",rows=matched))
    from .probes import subgroup_diagnostics
    csv_rows(output/"tables/subgroups.csv",subgroup_diagnostics(as_data(sample),REFERENCE,THRESHOLDS))
    del sample
    from .profile_report import geometry_cases, plot_geometry
    plot_geometry(output)
    geometry_cases(output,config["data_root"],REFERENCE)
    _atomic_json(output/"execution.json",dict(seconds=time.monotonic()-start,score_workers=workers,
                                              workers=resources,final_disk=host_disk(),
                                              sampling_increment={k:v for k,v in increment.items() if k!="rows"}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/home/jasongao/Data/STU"))
    parser.add_argument("--output", type=Path, default=Path("results/geometry"))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--stage", choices=("extract", "analyze", "all"), default="all")
    args = parser.parse_args()
    if not 1 <= args.workers <= len(os.sched_getaffinity(0)):
        parser.error("workers exceed the current CPU affinity")
    if args.stage in ("extract", "all"):
        extract(args.data_root, args.output, args.workers)
    if args.stage in ("analyze", "all"):
        analyze(args.output, min(args.workers,6))


if __name__ == "__main__":
    main()
