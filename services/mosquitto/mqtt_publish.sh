#!/bin/sh

# Define the pipe path
PIPE_PATH="/tmp/mqtt_pipe"

# Check if the named pipe already exists; create it if not
if [ ! -p "$PIPE_PATH" ]; then
  mkfifo "$PIPE_PATH"
fi

# Start the background process to generate data and write to the pipe
i=0
(
  while true; do
    x=$(awk -v val=$i 'BEGIN {print sin(val) + (rand() * 0.1)}')
    y=$(awk -v val=$i 'BEGIN {print sin(val + 1.57) + (rand() * 0.1)}')  # Phase shifted
    z=$(awk -v val=$i 'BEGIN {print sin(val + 3.14) + (rand() * 0.1)}')  # Another phase shift

    # Write the message to the pipe
    echo "{\"x\": $x, \"y\": $y, \"z\": $z}" > "$PIPE_PATH"

    # Increment i by 0.01 (sh doesn't support floating point arithmetic natively, using bc)
    i=$(echo "$i+0.001" | bc)

    # Generate data at 100Hz
    usleep 1000
  done
) &

# Use mosquitto_pub with -l to read continuously from the pipe and publish
tail -f "$PIPE_PATH" | mosquitto_pub -h mosquitto -t 'rattechan/sensor' -l