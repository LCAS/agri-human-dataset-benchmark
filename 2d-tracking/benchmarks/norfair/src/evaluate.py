import motmetrics as mm

def evaluate_mot(gt_path: str, pred_path: str):
    # Load files
    gt = mm.io.loadtxt(gt_path, fmt="mot15-2D", min_confidence=1)
    pred = mm.io.loadtxt(pred_path, fmt="mot15-2D", min_confidence=1)

    # Compare
    acc = mm.utils.compare_to_groundtruth(gt, pred, "iou", distth=0.5)

    # Compute metrics
    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=[
            "num_frames",
            "idf1",
            "idp",
            "idr",
            "precision",
            "recall",
            "mota", 
            "motp",
            "num_switches",
            "num_false_positives",
            "num_misses",
            "mostly_tracked",
            "mostly_lost",
        ],
        name="sequence"
    )

    # Pretty print
    print(mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    ))

    return summary

# Example usage:
evaluate_mot("D:/AOC/agri-human-dataset-benchmark/2d-tracking/benchmarks/norfair/outputs/out_vine_4swap+walk_st_ly_11_06_2024_2_label_gt_mot.txt", "D:/AOC/agri-human-dataset-benchmark/2d-tracking/benchmarks/norfair/outputs/out_vine_4swap+walk_st_ly_11_06_2024_2_label.txt")
