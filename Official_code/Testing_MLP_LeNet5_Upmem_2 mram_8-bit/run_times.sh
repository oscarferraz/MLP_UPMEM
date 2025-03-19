#!/bin/bash

# Output file name
OUTPUT_FILE="results.txt"

# Clear the output file before starting
> "$OUTPUT_FILE"

# Run the executable 5 times
for i in {1..5}; do
    # Execute the program and capture its output
    OUTPUT=$(./host)
    
    # Extract the line that contains the time information.
    # Expected line format:
    # L1 time (ms): 20.078000 L2 time (ms): 9.950000  L3 time (ms): 2.420000
    TIME_LINE=$(echo "$OUTPUT" | grep "L1 time (ms):")
    
    # Use awk to extract the times by their positions:
    # $4 => L1 time, $8 => L2 time, $12 => L3 time
    L1=$(echo "$TIME_LINE" | awk '{print $4}')
    L2=$(echo "$TIME_LINE" | awk '{print $8}')
    L3=$(echo "$TIME_LINE" | awk '{print $12}')
    
    # Append the extracted times to the output file (columns separated by space)
    echo "$L1 $L2 $L3" >> "$OUTPUT_FILE"

    sleep 0.5
done
