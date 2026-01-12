import csv
from pathlib import Path

def _parse_int(val, name):
    try:
        return int(val)
    except Exception:
        raise ValueError(f"{name} must be integer-like, got: {val}")

def merge_gt_csvs_unique(
    inputs,
    output_csv,
    output_fields=("path", "id", "cam", "color", "type"),
    make_cam_unique=True,
):
    merged_rows = []

    current_id_offset = 0
    current_cam_offset = 0

    for cfg in inputs:
        csv_path = Path(cfg["csv_path"])
        field_map = cfg["field_map"]
        path_prefix = cfg.get("path_prefix", "")

        if not csv_path.exists():
            print(f"[WARN] Missing GT file, skipping: {csv_path}")
            continue

        local_rows = []
        local_ids = set()
        local_cams = set()

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                out = {}

                for in_field, out_field in field_map.items():
                    out[out_field] = row[in_field]

                # parse + offset IDs
                vid = _parse_int(out["id"], "vehicleID") + current_id_offset
                cam = _parse_int(out["cam"].lstrip("c"), "cameraID")

                if make_cam_unique:
                    cam += current_cam_offset

                out["id"] = str(vid)
                out["cam"] = f"c{cam:03d}"

                if path_prefix:
                    out["path"] = str(Path(path_prefix) / out["path"])

                for f in output_fields:
                    out.setdefault(f, "")

                local_ids.add(vid)
                local_cams.add(cam)
                local_rows.append(out)

        # update offsets for next file
        if local_ids:
            current_id_offset = max(local_ids) + 1
        if make_cam_unique and local_cams:
            current_cam_offset = max(local_cams) + 1

        merged_rows.extend(local_rows)

    # write output
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(merged_rows)

    print(f"[OK] Written {len(merged_rows)} rows → {output_csv}")

if __name__ == "__main__":

    merge_gt_csvs_unique(
    inputs=[
        {
            "csv_path": "/home/tomass/tomass/data/VeRi/train.csv",
            "field_map": {
                "path": "path",
                "id": "id",
                "cam": "cam",
                "color": "color",
                "type": "type",
            },
            "path_prefix": "VeRi",
        },
        {
            "csv_path": "/home/tomass/tomass/data/VehicleX_VeRi/train.csv",
            "field_map": {
                "path": "path",
                "id": "id",
                "cam": "cam",
                "color": "color",
                "type": "type",
            },
            "path_prefix": "VehicleX_VeRi",
        },
    ],
    output_csv="/home/tomass/tomass/data/VeRi_VehicX_labels/train.csv",
    )