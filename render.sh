#!/bin/bash

file=$1

if [ -z "$file" ]; then
    echo "Usage: render.sh <file>"
    exit 1
fi

renderer=opengl
# renderer=cairo
manim \
    -pqm ${file} DynamicGraph --renderer=${renderer} \
    # --write_to_movie
