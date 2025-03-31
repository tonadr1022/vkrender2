#!/bin/bash
# thanks gpt :D

find_shader_files() {
	local directory="$1"
	find "$directory" -type f \( -name "*.vert.glsl" -o -name "*.frag.glsl" \)
}

compile_shader() {
	local file="$1"
	local spv_file="${file%.*}.spv"

	echo "Compiling: $file"
	if output=$(glslangValidator -V "$file" -o "$spv_file" -gVS 2>&1); then
		echo "$output"
	else
		echo "Error compiling $file:"
		echo "$output"
	fi
}

main() {
	local directory="$1"
	if [ -z "$directory" ]; then
		read -p "Enter the directory to search: " directory
	fi

	declare -a pids
	while IFS= read -r file; do
		compile_shader "$file" &
		pids+=("$!")
	done < <(find_shader_files "$directory")

	for pid in "${pids[@]}"; do
		wait "$pid"
	done
}

main "$1"
