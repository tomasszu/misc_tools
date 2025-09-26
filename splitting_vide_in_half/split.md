## Check lengths

```sh

ffmpeg -i input.mp4


```

## Get 2 halves

```sh

# First half (0–60s)
ffmpeg -i input.mp4 -t 60 -c copy output_part1.mp4

# Second half (from 60s to end)
ffmpeg -i input.mp4 -ss 60 -c copy output_part2.mp4



```