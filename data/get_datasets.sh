#!/usr/bin/env bash

PROG=$(basename "$0")

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)

error_msg() {
	echo -e "$PROG: $*" >&2
}

die() {
	local code="$1"
	shift
	error_msg "$@"
	exit $code
}

prereqs=(
	'kaggle',
)

for req in prereqs; do
	if [ -x "$(command -v $req)" ]; then
		die 1 "Prerequisite: $req is not installed."	
	fi
done

download_dir="$script_dir/HSMD"
if [ -d "$download_dir" ]; then
	die 1 "HSMD dataset already downloaded."
else
	mkdir -p "$download_dir"
	kaggle datasets download -d 'borismarjanovic/price-volume-data-for-all-us-stocks-etfs' -p "$download_dir" --unzip
	rm -rf "$download_dir/data"
fi