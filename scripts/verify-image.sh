#!/usr/bin/env bash
# Validate an ext4 image with e2fsprogs, natively on Linux or through WSL on
# Windows (Git Bash). Exits non-zero when e2fsck reports any problem.
#
#   scripts/verify-image.sh IMAGE [DEBUGFS_REQUEST ...]
#
# Each extra argument is passed to `debugfs -R` after the check, for example:
#   scripts/verify-image.sh out.img 'ls -l /opt' 'stat /opt/app/big.bin'
#
# On Windows, set EXT4KIT_WSL_DISTRO to choose the distribution; otherwise the
# first one with e2fsck installed is used.
set -euo pipefail

if [ $# -lt 1 ]; then
    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//' >&2
    exit 2
fi
image=$1
shift
[ -f "$image" ] || { echo "error: $image does not exist" >&2; exit 2; }

if command -v e2fsck >/dev/null 2>&1; then
    run() { "$@"; }
    target=$image
elif command -v wsl >/dev/null 2>&1; then
    # Stop Git Bash from rewriting arguments that look like paths.
    export MSYS_NO_PATHCONV=1
    distro=${EXT4KIT_WSL_DISTRO:-}
    if [ -z "$distro" ]; then
        for candidate in $(wsl -l -q | tr -d '\r\0'); do
            if wsl -d "$candidate" -e sh -c 'command -v e2fsck' >/dev/null 2>&1; then
                distro=$candidate
                break
            fi
        done
    fi
    [ -n "$distro" ] || {
        echo "error: no WSL distribution has e2fsck (install e2fsprogs)" >&2
        exit 2
    }
    run() { wsl -d "$distro" -e "$@"; }
    windows_path=$(cygpath -m "$(realpath "$image")")
    target=$(wsl -d "$distro" -e wslpath -a "$windows_path" | tr -d '\r')
else
    echo "error: e2fsck not found and WSL is unavailable" >&2
    exit 2
fi

# -f forces a full check of a clean filesystem; -n opens it read-only.
# Exit codes: 0 clean, 4 errors left uncorrected, 8+ operational failure.
status=0
run e2fsck -fn "$target" || status=$?

for request in "$@"; do
    echo "--- debugfs -R '$request'"
    run debugfs -R "$request" "$target" 2>&1 | grep -v '^debugfs [0-9]'
done

if [ "$status" -ne 0 ]; then
    echo "e2fsck FAILED with exit code $status" >&2
fi
exit "$status"
