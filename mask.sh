#!/bin/sh
SCRIPT="${0%.*}.py"
EXT="png"
INPUT="input.png"
OUTPUT="output-${SCRIPT%.*}"
MASK="m.png"

for I in $(seq 0 4); do
	for J in $(seq 0 3); do
		for F in $(seq 0 1); do
			python "${SCRIPT}" -i "${INPUT}" -m "${MASK}" -in ${I} -so ${J} -f ${F} -o "${OUTPUT}.in${I}so${J}f${F}.${EXT}"
		done
	done
done
