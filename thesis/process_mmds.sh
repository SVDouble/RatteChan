#!/bin/bash
# Usage: ./process_mmds.sh <file_or_directory>
# This script processes .mmd files: if a directory is given, it processes all .mmd files in it; if a file is given, it processes that file.

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <file_or_directory>"
  exit 1
fi

INPUT_PATH="$1"

process_file() {
  local mmd_file="$1"
  local base
  base=$(basename "$mmd_file" .mmd)
  local dir
  dir=$(dirname "$mmd_file")
  local tmp_file
  tmp_file=$(mktemp --suffix=.pdf)
  echo "Processing $mmd_file -> ${dir}/${base}.pdf"
  mmdc -f -i "$mmd_file" -o "$tmp_file" && \
  pdfcrop "$tmp_file" "${dir}/${base}.pdf" && \
  rm "$tmp_file"
}

export -f process_file

if [ -d "$INPUT_PATH" ]; then
  find "$INPUT_PATH" -maxdepth 1 -type f -name "*.mmd" | parallel process_file {}
elif [ -f "$INPUT_PATH" ]; then
  process_file "$INPUT_PATH"
else
  echo "Error: $INPUT_PATH is not a valid file or directory."
  exit 1
fi
