# Code launch parameters for the different videos

## CityFlow vid4

```py

    # Args concerning the establishment of crop zones for video
    parser.add_argument('--crop_zone_rows_vid1', type=int, default=7, help='Number of rows in the crop zone grid for the first video.')
    parser.add_argument('--crop_zone_cols_vid1', type=int, default=6, help='Number of columns in the crop zone grid for the first video.')
    parser.add_argument('--crop_zone_area_bottom_left_vid1', type=tuple, default=(0, 1000), help='Bottom-left corner of the crop zone area as a tuple (x, y) for the first video.')
    parser.add_argument('--crop_zone_area_top_right_vid1', type=tuple, default=(1750, 320), help='Top-right corner of the crop zone area as a tuple (x, y) for the first video.')


```

