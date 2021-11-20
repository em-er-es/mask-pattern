#!/bin/bash
if ! command -v montage &>/dev/null; then
	echo "Not able to find \`montage\`. ImageMagick probably not installed."
	exit 2
fi
montage -resize 25% output-mask.* -tile 4x -geometry +0+0 "${0%.*}.png"
