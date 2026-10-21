#!/usr/bin/env bash
# Server tests, one process per test module, in parallel (unittest discover runs the 84 modules one by one: ~45 min).
#
#   tools/test.sh                 every module but the slow end-to-end ones (SLOW), JOBS at once: a few minutes
#   tools/test.sh --all           every module (the slow ones too; timing tests may need fewer JOBS)
#   tools/test.sh grading dsm     only these modules (test_ prefix optional)
#   JOBS=4 tools/test.sh          fewer at once (the timing tests: test_colormatch per_tile_cost, test_credits session)
#
# Prints one line per failing module and its FAIL / ERROR lines; logs in $TMPDIR/bne-tests/<module>.log.
set -u
cd "$(dirname "$0")/../server" || exit 2
PY="${PYTHON:-../.venv/bin/python}"
[ -x "$PY" ] || PY=python3
JOBS="${JOBS:-8}"
SLOW="test_integration_drive test_server test_e2e test_restart test_protocol_agreement"
LOG="${TMPDIR:-/tmp}/bne-tests"
rm -rf "$LOG" && mkdir -p "$LOG"

mods=()
if [ "${1:-}" = "--all" ]; then
    for f in tests/test_*.py; do mods+=("$(basename "$f" .py)"); done
elif [ $# -gt 0 ]; then
    for m in "$@"; do m="${m%.py}"; m="${m#tests/}"; [ "${m#test_}" = "$m" ] && m="test_$m"; mods+=("$m"); done
else
    for f in tests/test_*.py; do
        m="$(basename "$f" .py)"
        case " $SLOW " in *" $m "*) continue ;; esac
        mods+=("$m")
    done
fi

start=$(date +%s)
printf '%s\n' "${mods[@]}" | xargs -P "$JOBS" -I{} sh -c \
    '"$0" -m unittest discover -s tests -p "{}.py" > "$1/{}.log" 2>&1; echo "$? {}" >> "$1/_status"' "$PY" "$LOG"

fail=0
while read -r code m; do
    if [ "$code" != 0 ]; then
        fail=$((fail + 1))
        echo "FAILED $m"
        grep -E "^(FAIL|ERROR):" "$LOG/$m.log" | sed 's/^/    /'
    fi
done < <(sort -k2 "$LOG/_status")
echo "${#mods[@]} modules, $fail failing, $(( $(date +%s) - start )) s (logs: $LOG)"
[ "$fail" = 0 ]
