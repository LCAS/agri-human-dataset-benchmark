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
            "mota", "motp",
            "idf1",
            "num_switches",
            "num_false_positives", "num_misses",
            "mostly_tracked", "mostly_lost",
            "precision", "recall"
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
evaluate_mot("gt.txt", "tracking.txt")
