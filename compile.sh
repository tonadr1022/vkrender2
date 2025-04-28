#!/bin/bash
# thanks gpt :D

find_shader_files() {
	local directory="$1"
	find "$directory" -type f \( -name "*.vert.glsl" -o -name "*.frag.glsl" -o -name "*.comp.glsl" \)
}

compile_shader() {
	local file="$1"
	local debug="$2"
	local spv_file="${file%.*}.spv"

	echo "Compiling: $file"

	local flags="-V"
	if [ "$debug" = true ]; then
		flags="-gVS -V"
	fi

	if output=$(glslangValidator $flags "$file" -o "$spv_file" 2>&1); then
		echo "$output"
	else
		echo "Error compiling $file:"
		echo "$output"
	fi
}

main() {
	local debug=false
	local directory=

	# Parse options
	while getopts "d" opt; do
		case $opt in
		d) debug=true ;;
		*)
			echo "Usage: $0 [-d] [directory]"
			exit 1
			;;
		esac
	done
	shift $((OPTIND - 1))

	# Get directory argument
	directory="$1"
	if [ -z "$directory" ]; then
		read -p "Enter the directory to search: " directory
	fi

	declare -a pids
	while IFS= read -r file; do
		compile_shader "$file" "$debug" &
		pids+=("$!")
	done < <(find_shader_files "$directory")

	for pid in "${pids[@]}"; do
		wait "$pid"
	done
}

main "$@"
