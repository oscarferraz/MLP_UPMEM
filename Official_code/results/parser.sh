#!/bin/bash
set -euo pipefail

# Find all unique directories that contain at least one .json file.
# We use 'find' and then extract the parent directory (%h) of each .json file.
dirs=$(find . -type f -name "*.json" -printf "%h\n" | sort -u)

# Loop over each directory
for folder in $dirs; do
    echo "Processing folder: $folder"
    
    output_file="$folder/execution_times.csv"
    
    # Write CSV header into the output file
    echo "dpu_push_xfer_1 (ms),dpu_push_xfer_2 (ms),dpu_launch_1 (ms),dpu_push_xfer_3 (ms),dpu_push_xfer_4 (ms),dpu_push_xfer_5 (ms),dpu_launch_2 (ms),dpu_push_xfer_6 (ms),dpu_push_xfer_7 (ms),dpu_push_xfer_8 (ms),dpu_launch_3 (ms),dpu_push_xfer_9 (ms)" > "$output_file"
    
    # Loop through all .json files in this folder (sorted by name)
    for jsonfile in "$folder"/*.json; do
        echo "  Processing file: $jsonfile"
        
        # Initialize arrays to store durations (in ms) and the event stack
        push_durations=()
        launch_durations=()
        stack=()
        
        # Process each event in the JSON file
        while IFS= read -r line; do
            # Extract event phase, timestamp, and name (if present)
            ph=$(echo "$line" | jq -r '.ph')
            ts=$(echo "$line" | jq -r '.ts')
            name=$(echo "$line" | jq -r '.name // ""')
    
            if [ "$ph" = "B" ]; then
                # Push begin event info as "name|timestamp" onto the stack
                stack+=( "$name|$ts" )
            elif [ "$ph" = "E" ]; then
                # Pop the last begin event from the stack if available
                if [ ${#stack[@]} -gt 0 ]; then
                    top="${stack[-1]}"
                    unset 'stack[-1]'
                    event_name="${top%%|*}"
                    event_ts="${top##*|}"
                    # Compute duration in milliseconds for our events of interest
                    if [ "$event_name" = "dpu_push_xfer" ]; then
                        duration=$(echo "scale=6; ($ts - $event_ts) / 1000" | bc -l)
                        push_durations+=( "$duration" )
                    elif [ "$event_name" = "dpu_launch" ]; then
                        duration=$(echo "scale=6; ($ts - $event_ts) / 1000" | bc -l)
                        launch_durations+=( "$duration" )
                    fi
                fi
            fi
        done < <(jq -c '.traceEvents[]' "$jsonfile")
    
        # Check if we found the expected number of events
        if [ "${#push_durations[@]}" -ne 9 ] || [ "${#launch_durations[@]}" -ne 3 ]; then
            echo "    Warning: In $jsonfile, found ${#push_durations[@]} dpu_push_xfer and ${#launch_durations[@]} dpu_launch events." >&2
            continue
        fi
    
        # Arrange the values in the specified order:
        # dpu_push_xfer_1, dpu_push_xfer_2, dpu_launch_1, dpu_push_xfer_3, dpu_push_xfer_4,
        # dpu_push_xfer_5, dpu_launch_2, dpu_push_xfer_6, dpu_push_xfer_7, dpu_push_xfer_8,
        # dpu_launch_3, dpu_push_xfer_9
        row="${push_durations[0]},${push_durations[1]},${launch_durations[0]},${push_durations[2]},${push_durations[3]},${push_durations[4]},${launch_durations[1]},${push_durations[5]},${push_durations[6]},${push_durations[7]},${launch_durations[2]},${push_durations[8]}"
    
        # Append the row to the CSV file
        echo "$row" >> "$output_file"
    done
    
    echo "Finished processing folder: $folder (CSV: $output_file)"
done

echo "All folders processed."
