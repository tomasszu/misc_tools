import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
import numpy as np


def load_gt(xml_path, path_root="image_train"):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    items_node = root.find("Items")
    if items_node is None:
        raise ValueError("No <Items> element found.")

    rows = []
    for item in items_node.findall("Item"):
        rows.append({
            "path": f"{path_root}/{item.get('imageName')}",
            "id": item.get("vehicleID"),
            "cam": item.get("cameraID"),
            "color": item.get("colorID"),
            "type": item.get("typeID"),
        })
    return rows


def print_dataset_stats(rows, top_k=5):
    # ---------- basic counts ----------
    ids = [r["id"] for r in rows]
    cams = [r["cam"] for r in rows]

    id_counts = Counter(ids)
    cam_counts = Counter(cams)

    counts = np.array(list(id_counts.values()))

    print("\n=== DATASET OVERVIEW ===")
    print(f"Total images: {len(rows)}")
    print(f"Total unique vehicle IDs: {len(id_counts)}")
    print(f"Total cameras: {len(cam_counts)}")

    # ---------- per-ID image stats ----------
    print("\n=== IMAGES PER VEHICLE ID ===")
    print(f"Mean: {counts.mean():.2f}")
    print(f"Std:  {counts.std():.2f}")
    print(f"Min:  {counts.min()}")
    print(f"Max:  {counts.max()}")

    # ---------- top / bottom IDs ----------
    sorted_ids = id_counts.most_common()

    print(f"\nTop {top_k} vehicle IDs by image count:")
    for vid, c in sorted_ids[:top_k]:
        print(f"  ID {vid}: {c}")

    print(f"\nBottom {top_k} vehicle IDs by image count:")
    for vid, c in sorted_ids[-top_k:]:
        print(f"  ID {vid}: {c}")

    max_id, max_count = sorted_ids[0]
    print(f"\nVehicle ID with max images: {max_id} ({max_count} images)")

    # ---------- camera distribution ----------
    print("\n=== CAMERA DISTRIBUTION ===")
    total_imgs = len(rows)
    for cam, c in cam_counts.most_common():
        pct = 100.0 * c / total_imgs
        print(f"Camera {cam}: {c} images ({pct:.2f}%)")

    # ---------- camera × vehicle sanity ----------
    cam_per_id = defaultdict(set)
    for r in rows:
        cam_per_id[r["id"]].add(r["cam"])

    cams_per_vehicle = Counter(len(v) for v in cam_per_id.values())

    print("\n=== CAMERAS PER VEHICLE ID ===")
    for n_cams in sorted(cams_per_vehicle):
        print(f"{n_cams} cameras: {cams_per_vehicle[n_cams]} vehicle IDs")

if __name__ == "__main__":
    xml_path = "/home/tomass/tomass/data/VeRi/train_label.xml"
    rows = load_gt(xml_path)
    print_dataset_stats(rows)
