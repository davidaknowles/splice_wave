#!/bin/bash

module load macs2

set -e

# Directory containing the .bw files
input_dir="$1"

# Loop over each .bw file in the input directory
for bw_file in "$input_dir"/*.bw; do
    echo $bw_file
    
    # Extract the base name of the file (without extension)
    base_name=$(basename "$bw_file" .bw)
    
    # Define the output bedgraph file path
    bedgraph_file="${input_dir}/${base_name}.bedgraph"
    
    # Convert the .bw file to .bedgraph
    bigWigToBedGraph "$bw_file" "$bedgraph_file"

    narrowpeak_file=${input_dir}/${base_name}.narrow_peak
    
    # Run MACS2 on the .bedgraph file
    macs2 bdgpeakcall -i "$bedgraph_file" --ofile $narrowpeak_file

    gzip $narrowpeak_file

    #rm $bedgraph_file $bw_file
done
