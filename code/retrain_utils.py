"""
Shared helper for loading pre-existing retrain checkpoints.

Filename pattern:
    {Model}-{Dataset}-seed{seed}-ret{perc}-{epoch}-best.pth

If multiple files exist for the same seed, the one with the highest epoch is used.
"""
import glob
import os
import re


def find_best_retrain_model(retrain_folder, model_name, dataset_name, seed, ret_perc=None):
    """
    Find the best pre-existing retrain model for a given seed.

    Search order:
      1. retrain_folder/Retrain/          (full retrain; ret0 or ret100)
      2. retrain_folder/{ret_perc}/       (25 / 50 / 75 retain subsets)
      3. retrain_folder/                  (files placed directly in the folder)

    Among matches, pick the file with the highest epoch in `...-{epoch}-best.pth`.
    Returns the path, or None if nothing is found.
    """
    if not retrain_folder:
        return None

    search_paths = []
    if ret_perc is None or int(ret_perc) in (0, 100):
        search_paths.append(os.path.join(retrain_folder, "Retrain"))
    if ret_perc is not None and int(ret_perc) not in (0, 100):
        search_paths.append(os.path.join(retrain_folder, str(int(ret_perc))))
    search_paths.append(retrain_folder)
    retrain_sub = os.path.join(retrain_folder, "Retrain")
    if retrain_sub not in search_paths:
        search_paths.append(retrain_sub)

    matches = []
    seen = set()
    for search_path in search_paths:
        if not os.path.exists(search_path):
            continue
        pattern = os.path.join(
            search_path, f"{model_name}-{dataset_name}-seed{seed}-ret*-*-best.pth"
        )
        for path in glob.glob(pattern):
            if path not in seen:
                seen.add(path)
                matches.append(path)

    if not matches:
        return None

    def parse_ret_epoch(path):
        basename = os.path.basename(path)
        match = re.search(r"-ret(\d+)-(\d+)-best\.pth$", basename)
        if match:
            return int(match.group(1)), int(match.group(2))
        return None, 0

    filtered = []
    for path in matches:
        ret_val, epoch = parse_ret_epoch(path)
        if ret_val is None:
            continue
        if ret_perc is not None:
            wanted = int(ret_perc)
            if wanted in (0, 100):
                # ResNet Cifar 10 uses ret0; other datasets use ret100
                if ret_val not in (0, 100):
                    continue
            elif ret_val != wanted:
                continue
        filtered.append((epoch, path))

    if not filtered:
        filtered = [(parse_ret_epoch(p)[1], p) for p in matches]

    best_path = max(filtered, key=lambda x: x[0])[1]
    return best_path
