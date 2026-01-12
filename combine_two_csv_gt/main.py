import csv
from pathlib import Path


def merge_gt_csvs(
    inputs,
    output_csv,
    output_fields=("path", "id", "cam", "color", "type"),
):
    """
    inputs: list of dicts, each with:
        {
            "csv_path": str | Path,
            "field_map": dict,        # input_field -> output_field
            "path_prefix": str | None # prepended to path
        }

    output_csv: str | Path
    output_fields: tuple[str]
    """

    merged_rows = []

    for cfg in inputs:
        csv_path = Path(cfg["csv_path"])
        field_map = cfg["field_map"]
        path_prefix = cfg.get("path_prefix", "")

        if not csv_path.exists():
            print(f"[WARN] Missing GT file, skipping: {csv_path}")
            continue

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                out_row = {}

                for in_field, out_field in field_map.items():
                    if in_field not in row:
                        raise KeyError(
                            f"Field '{in_field}' not found in {csv_path}"
                        )
                    out_row[out_field] = row[in_field]

                # extend path if requested
                if "path" in out_row and path_prefix:
                    out_row["path"] = str(
                        Path(path_prefix) / out_row["path"]
                    )

                # ensure all output fields exist
                for f in output_fields:
                    out_row.setdefault(f, "")

                merged_rows.append(out_row)

    # write merged CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(merged_rows)

    print(f"[OK] Written {len(merged_rows)} rows → {output_csv}")

if __name__ == "__main__":

# two files
    # merge_gt_csvs(
    # inputs=[
    #     {
    #         "csv_path": "/home/tomass/tomass/data/VeRi/train.csv",
    #         "field_map": {
    #             "path": "path",
    #             "id": "id",
    #             "cam": "cam",
    #             "color": "color",
    #             "type": "type",
    #         },
    #         "path_prefix": "VeRi",
    #     },
    #     {
    #         "csv_path": "/home/tomass/tomass/data/VehicleX_VeRi/train.csv",
    #         "field_map": {
    #             "path": "path",
    #             "id": "id",
    #             "cam": "cam",
    #             "color": "color",
    #             "type": "type",
    #         },
    #         "path_prefix": "VehicleX_VeRi",
    #     },
    # ],
    # output_csv="/home/tomass/tomass/data/VeRi_VehicX/train.csv",
    # )

#just one file used function to edit

    merge_gt_csvs(
    inputs=[
        {
            "csv_path": "/home/tomass/tomass/data/VeRi/val.csv",
            "field_map": {
                "path": "path",
                "id": "id",
                "cam": "cam",
                "color": "color",
                "type": "type",
            },
            "path_prefix": "VeRi",
        }
    ],
    output_csv="/home/tomass/tomass/data/VeRi_VehicX_labels/val.csv",
    )