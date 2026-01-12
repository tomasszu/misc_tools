import xml.etree.ElementTree as ET
import csv

def xml_to_csv(xml_path, csv_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Find the <Items> node
    items_node = root.find("Items")
    if items_node is None:
        raise ValueError("No <Items> element found.")
    
    path_root = "VeRi_ReID_Simulation"

    rows = []
    for item in items_node.findall("Item"):
        rows.append({
            "path": str(f"{path_root}/{item.get('imageName')}"),
            "id": item.get("vehicleID"),
            "cam": item.get("cameraID"),
            "color": item.get("colorID"),
            "type": item.get("typeID"),
        })

    # Write to CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["path", "id", "cam", "color", "type"]
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    xml_to_csv("/home/tomass/tomass/data/VehicleX_VeRi/VeRi_label.xml", "/home/tomass/tomass/data/VehicleX_VeRi/train.csv")
