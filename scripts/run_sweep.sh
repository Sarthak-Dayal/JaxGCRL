#!/usr/bin/env bash
# Run a list of commands, keeping every attempt's logs and crash artifacts.
#
#   ./scripts/run_sweep.sh SPEC [SWEEP_NAME]      # commands from a file
#   ... | ./scripts/run_sweep.sh - [SWEEP_NAME]   # commands from stdin, for generated sweeps
#
# SPEC is one shell command per line, exactly as you would type it. Blank lines and `#` comments
# are ignored (a `#` must start the line -- mid-line it belongs to the command). An optional
# `name :: ` prefix names the run; otherwise runs are numbered.
#
#   # a comment
#   python run.py --env ant --seed 1 hiql --low-alpha 3.0
#   crl_ant_hi_lr :: python run.py --env ant --seed 1 crl --policy-lr 1e-3
#   taskset -c 0-7,12-31 python run.py --env ant crl
#
# This script knows nothing about agents, environments or hyperparameters. It runs commands,
# retries ones that die from a flaky signal, and files the output. Anything that has to decide
# *what* to run -- per-env episode lengths, seeds, wandb tags -- belongs in whatever generates
# the spec.
#
# Layout, one directory per attempt so a retry never overwrites the evidence from the failure:
#
#   experiments/<sweep>/
#     spec.txt          the spec verbatim, so the sweep is reproducible from its own directory
#     sweep.meta        git commit, dirtiness, host, date, resolved settings
#     manifest.tsv      run, status, attempts, seconds, exit code
#     <run>/
#       cmd.txt         the command
#       attempt_01/
#         stdout.log
#         exit_code
#         crash.txt     backtrace from coredumpctl, when the run dumped core
#         xla.tar.zst   HLO/PTX dump: before/after-optimization HLO, LLVM IR, PTX, autotune
#                       results. ~150MB raw, ~10MB compressed. XLA_DUMP=0 disables.

set -uo pipefail

SPEC=${1:-}
[[ -z "$SPEC" ]] && { sed -n '2,31p' "$0" | sed 's/^# \?//'; exit 1; }

SWEEP_NAME=${2:-sweep-$(date +%Y-%m-%d_%H-%M)}
EXPERIMENTS=${EXPERIMENTS:-experiments}
SWEEP_DIR="$EXPERIMENTS/$SWEEP_NAME"
MAX_ATTEMPTS=${MAX_ATTEMPTS:-2}
# Exit codes worth retrying. 139 is SIGSEGV and 132 is SIGILL; we see both intermittently at
# varied, unrelated points inside the compiler, so they are not reproducible and a retry usually
# succeeds. See experiments/README or the crash notes for why they happen at all.
RETRY_CODES=${RETRY_CODES:-"139 132"}

mkdir -p "$SWEEP_DIR"
if [[ "$SPEC" == "-" ]]; then cat > "$SWEEP_DIR/spec.txt"; else cp "$SPEC" "$SWEEP_DIR/spec.txt"; fi

{
  echo "sweep:        $SWEEP_NAME"
  echo "started:      $(date -Is)"
  echo "host:         $(hostname)"
  echo "commit:       $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "dirty:        $(if git diff --quiet 2>/dev/null; then echo no; else echo yes; fi)"
  echo "max_attempts: $MAX_ATTEMPTS   retry_codes: $RETRY_CODES"
} > "$SWEEP_DIR/sweep.meta"

printf 'run\tstatus\tattempts\tseconds\texit\n' > "$SWEEP_DIR/manifest.tsv"

save_crash() {  # save_crash <pid> <dest>
  local pid=$1 dest=$2
  command -v coredumpctl >/dev/null 2>&1 || return 0
  coredumpctl info "$pid" 2>/dev/null | sed -n '/Stack trace/,$p' | head -60 > "$dest" || true
  [[ -s "$dest" ]] || rm -f "$dest"
}

total=0; failed=0
while IFS= read -r line || [[ -n "$line" ]]; do
  line="$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
  # Whole-line comments only. A `#` mid-line belongs to the command -- stripping from the first
  # `#` anywhere truncates things like `echo $#` and leaves a command that "succeeds" silently.
  [[ -z "$line" || "$line" == \#* ]] && continue
  total=$((total + 1))

  if [[ "$line" == *" :: "* ]]; then
    name="${line%% :: *}"; cmd="${line#* :: }"
  else
    # Position for reading order, plus a hash of the command so a reordered spec still lands in
    # a directory you can match back to its line.
    cmd="$line"
    name=$(printf 'run_%03d_%s' "$total" "$(printf '%s' "$line" | cksum | cut -d' ' -f1 | tail -c 5)")
  fi

  run_dir="$SWEEP_DIR/$name"; mkdir -p "$run_dir"
  # `eval` so quoted arguments in the spec survive as single words.
  eval "argv=($cmd)"
  printf '%s\n' "$cmd" > "$run_dir/cmd.txt"

  attempt=1; rc=0; started=$SECONDS
  while true; do
    adir=$(printf '%s/attempt_%02d' "$run_dir" "$attempt"); mkdir -p "$adir"

    # Dump the HLO/PTX for every attempt. Raw it is ~150MB per run, but it is nearly all text and
    # compresses ~15x, so the kept cost is ~10MB per run. Set XLA_DUMP=0 to turn it off.
    if [[ "${XLA_DUMP:-1}" == "1" ]]; then
      export XLA_FLAGS="--xla_dump_to=$adir/xla ${XLA_FLAGS_BASE:-}"
    else
      export XLA_FLAGS="${XLA_FLAGS_BASE:-}"
    fi

    echo "[$(date +%H:%M:%S)] $name (attempt $attempt)"
    "${argv[@]}" > "$adir/stdout.log" 2>&1 &
    pid=$!
    wait $pid; rc=$?
    echo "$rc" > "$adir/exit_code"

    if [[ -d "$adir/xla" ]]; then
      tar -C "$adir" -caf "$adir/xla.tar.zst" xla 2>/dev/null && rm -rf "$adir/xla"
    fi

    [[ $rc -eq 0 ]] && break
    save_crash "$pid" "$adir/crash.txt"
    [[ " $RETRY_CODES " == *" $rc "* && $attempt -lt $MAX_ATTEMPTS ]] || break
    echo "  retrying after exit $rc"
    attempt=$((attempt + 1))
  done
  unset XLA_FLAGS

  if [[ $rc -eq 0 ]]; then st=ok; else st=FAILED; failed=$((failed + 1)); fi
  printf '%s\t%s\t%d\t%d\t%d\n' "$name" "$st" "$attempt" "$((SECONDS - started))" "$rc" \
    >> "$SWEEP_DIR/manifest.tsv"
  echo "  -> $st (exit $rc, $((SECONDS - started))s)"
done < <(if [[ "$SPEC" == "-" ]]; then cat "$SWEEP_DIR/spec.txt"; else cat "$SPEC"; fi)

echo "finished: $((total - failed))/$total ok -- $SWEEP_DIR"
[[ $failed -eq 0 ]]
