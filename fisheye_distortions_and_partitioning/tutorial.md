Here’s a compact, working toolkit that:

    adds fisheye/barrel-style distortion (radial polynomial),

    partitions frames into a 3×2 grid and letterboxes each to 640×640 (YOLO-style),

    maps bbox coordinates through the distortion and into each tile.



What you get

    00_base_with_gt.png: synthetic frame with 3 “vehicle” boxes (green).

    01_distorted_with_mapped_boxes.png: same frame after distortion, boxes remapped (red).

    tile_*.png: 6 tiles (3×2 grid) at 640×640 with per-tile boxes drawn (white).

    manifest.json: per-tile metadata (region_xyxy in the global image + mapped_bboxes_xyxy in tile coords).

How to use on your data

    Replace the synthetic generator with your real images + bboxes (or just images if you only want crops).

    Tweak DistortionParams(k1,k2,k3,...) to match how “fisheye” you want it. Negative k1 → barrel/fisheye.

Key functions (read them in the code cell)

    distort_image(img, params): applies polynomial distortion.

    map_bboxes_through_distortion(bboxes_xyxy, img_size, params): transforms x1y1x2y2 via corner mapping.

    partition_image(img, grid=(3,2), tile_out=(640,640)): splits, letterboxes each tile.

    remap_bbox_to_tile(bbox_xyxy, region_xyxy, tile_out): global bbox → per-tile letterboxed coords.

If you want, I can extend this to:

    load YOLO TXT bboxes and emit updated TXT files per tile,

    simulate equirectangular 360 → multi-perspective “virtual cameras” instead of a flat grid,

    add Bytetrack-style ID carry-over compatibility by keeping a mapping from global IDs → tile-local detections.