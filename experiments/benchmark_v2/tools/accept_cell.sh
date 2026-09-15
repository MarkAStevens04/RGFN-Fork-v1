#!/bin/bash
# The ONLY sanctioned path from "a cell exists" to "a cell is done". Verify, back up, verify the
# backup, record it, freeze.
#
# WHY THIS EXISTS. `experiments/benchmark_v2/README.md` says "As soon as a stage passes verification,
# back that stage up ... per cell, per stage -- not per phase." That was a paragraph of prose, and a
# correct script nobody calls is not a backup. Prose guards have failed this project twice in one
# week: an allowlist that copied 0.7% of its payload and exited 0, and a `--sizes-only` check whose
# text refused to say "verified" while its exit code said it anyway. This makes the backup a
# PRECONDITION of the terminal state rather than a thing to remember.
#
# THE ORDER IS FORCED, not chosen. Two facts collide: /project is mounted on LOGIN nodes only, so the
# backup cannot live inside anything that may run on a compute node; and `.verified.json` lives inside
# train/, which freeze chmods a-w, so nothing can be written there afterwards. Hence:
#
#     verify -> back up -> verify the backup -> record in the ledger -> freeze
#
# The backup cannot come after the freeze because there would be nowhere left to record that it
# happened. And the record lives in the LEDGER, not in `.verified.json`, because the ledger is outside
# the frozen directory, is append-only, and already declares a `backup` stage ("a copy landed in
# /project") that nothing had used yet.
#
# THE FAILURE MODE IS THE POINT. A cell whose backup did not run, or ran and covered less than it
# should, fails here and never freezes -- so it reads exactly like a cell that failed its checks.
# There is no state in which a cell looks done without a verified backup behind it, because the only
# writer of the terminal state is the script that does the backup.
#
# NOT A REPLACEMENT FOR `scripts/backup_scratch_critical.sh`. That is the periodic WHOLESALE sweep
# over the entire v2 tree; this keeps ONE cell current at the moment it is accepted. Calling the
# wholesale script once per cell would be 54 full sweeps. They coexist because rsync is idempotent,
# and the sweep still matters: it catches anything that reached the tree outside this path.
#
# Usage:
#   experiments/benchmark_v2/tools/accept_cell.sh scent/seh/42 [--arm a]
#   experiments/benchmark_v2/tools/accept_cell.sh --all [--arm a]
#   experiments/benchmark_v2/tools/accept_cell.sh scent/seh/42 --dry-run
#
# Exit 0 = accepted (verified, backed up, backup content-verified, recorded, frozen).
# Non-zero = refused at some step, and the step is named.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python}"

# Where tier 1b lands the tree: `rsync "$V2_SRC/" "$DEST/v2/"`. Same default, one definition.
BACKUP_ROOT="${BENCHMARK_V2_BACKUP:-/project/def-naeilum/naeilum/SDL3/RGFN_LSD_MarkStevens/backups/scratch_runs/rgfn_runs/v2}"

ARM=a
ALL=0
CELL=""
DRY=0

while [ $# -gt 0 ]; do
    case "$1" in
        --arm) ARM="$2"; shift 2 ;;
        --all) ALL=1; shift ;;
        --dry-run) DRY=1; shift ;;
        --backup-root) BACKUP_ROOT="$2"; shift 2 ;;
        -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) CELL="$1"; shift ;;
    esac
done

if [ "$ALL" = 0 ] && [ -z "$CELL" ]; then
    echo "usage: accept_cell.sh GEN/TARGET/SEED [--arm a] [--dry-run]" >&2
    echo "       accept_cell.sh --all [--arm a]" >&2
    exit 2
fi

# -- /project is a LOGIN-NODE mount. Fail loudly here rather than letting a compute-node driver
# silently skip the backup and call the cell done. A driver inside a SLURM job should leave the cell
# verified-but-unaccepted and let a login-node sweep finish it.
if [ ! -d "$(dirname "$BACKUP_ROOT")" ]; then
    echo "REFUSED: $BACKUP_ROOT is not reachable." >&2
    echo "  /project is mounted on the LOGIN nodes, not on compute. Acceptance is a login-node" >&2
    echo "  operation by construction: it has to copy the cell off scratch before freezing it." >&2
    echo "  If you are inside a SLURM job, do NOT skip this -- leave the cell unaccepted and run" >&2
    echo "  'accept_cell.sh --all' from a login node afterwards." >&2
    exit 3
fi

# ---------------------------------------------------------------------------------------------
accept_one() {
    local cell="$1"
    local gen tgt seed rest train_dir tag dest

    gen="${cell%%/*}"; rest="${cell#*/}"; tgt="${rest%%/*}"; seed="${rest##*/}"
    tag="${gen}_${tgt}_s${seed}"
    dest="$BACKUP_ROOT/train/$tag/arm$ARM"

    train_dir=$("$PY" "$HERE/manifest.py" --emit "$gen" "$tgt" "$seed" --arm "$ARM" 2>/dev/null \
                | sed -n 's/^TRAIN_DIR=//p' | tr -d "'")
    if [ -z "$train_dir" ] || [ ! -d "$train_dir" ]; then
        echo "REFUSED $cell arm$ARM: no train dir (not in grid.csv, or nothing trained yet)" >&2
        return 1
    fi

    # -- 1. the checks. Unchanged from freeze's own gate; a read-only wrapper around a broken
    # artifact just looks finished, and so does a backup of one.
    if ! "$PY" "$HERE/verify_cell.py" --cell "$cell" --arm "$ARM" --stage train \
            >/tmp/accept_$$.log 2>&1; then
        echo "REFUSED $cell arm$ARM: verification failed -- not backed up, not frozen" >&2
        sed -n '/FAIL/p' /tmp/accept_$$.log | sed 's/^/    /' >&2
        rm -f /tmp/accept_$$.log
        return 1
    fi
    rm -f /tmp/accept_$$.log

    if [ "$DRY" = 1 ]; then
        echo "would accept $cell arm$ARM: $train_dir -> $dest"
        return 0
    fi

    # -- 2. the backup. --chmod KEEPS THE DESTINATION WRITABLE: `-a` implies `-p`, so a frozen
    # (read-only) source would otherwise land read-only and the NEXT incremental sync could not write
    # into its own destination. Freeze protects the live tree, not the copy.
    mkdir -p "$dest"
    if ! rsync -a --chmod=Du+w,Fu+w "$train_dir/" "$dest/"; then
        echo "REFUSED $cell arm$ARM: rsync failed -- not frozen" >&2
        return 1
    fi

    # -- 3. did the backup actually land, and is it the right bytes? EXIT 0 IS REQUIRED, not "not 1".
    # A structural-only pass returns 2 and is a refusal here: accepting on it would make acceptance
    # cheap and meaningless, which is the whole reason exit 2 exists.
    "$PY" "$HERE/verify_backup.py" --cell "$cell" --arm "$ARM" --backup-root "$BACKUP_ROOT" \
        >/tmp/accept_bk_$$.log 2>&1
    local bk_rc=$?
    if [ "$bk_rc" -ne 0 ]; then
        echo "REFUSED $cell arm$ARM: backup verification returned $bk_rc (0 required) -- not frozen" >&2
        sed -n '/MISSING\|MISMATCH\|CONTENT WAS NOT CHECKED/p' /tmp/accept_bk_$$.log \
            | sed 's/^/    /' >&2
        rm -f /tmp/accept_bk_$$.log
        return 1
    fi
    rm -f /tmp/accept_bk_$$.log

    # -- 4. record it. source_path is the V2 cell, not the /project copy, deliberately: `ledger verify`
    # re-hashes source_path, so this row becomes a drift check on the frozen v2 cell. Whether the
    # /project copy is still good is verify_backup.py's question and it reads only the backup.
    if ! "$PY" "$HERE/ledger.py" record --stage backup --cell "$cell" --arm "$ARM" \
            --origin copied --source "$train_dir" \
            --note "backed up to $dest; content-verified against .copy_manifest.json" >/dev/null; then
        echo "REFUSED $cell arm$ARM: ledger record failed -- not frozen" >&2
        return 1
    fi

    # -- 5. freeze. Last, because it makes train/ read-only and nothing can be added afterwards.
    if ! bash "$HERE/freeze_cell.sh" "$cell" --arm "$ARM" >/dev/null 2>&1; then
        echo "REFUSED $cell arm$ARM: freeze failed AFTER a good backup -- the backup stands, the" >&2
        echo "    cell is not frozen. Re-run accept_cell.sh; steps 2-4 are idempotent." >&2
        return 1
    fi

    echo "ACCEPTED $cell arm$ARM  verified, backed up, content-verified, recorded, frozen"
    return 0
}

# ---------------------------------------------------------------------------------------------
rc=0
if [ "$ALL" = 1 ]; then
    while read -r tag; do
        [ -z "$tag" ] && continue
        # Same split as freeze_cell.sh, character for character. tag is <gen>_<target>_s<seed>, and
        # recovering the triple by stripping the KNOWN gen prefix avoids guessing where the target
        # ends -- a second, subtly different copy of this is how the two scripts would drift apart.
        seed="${tag##*_s}"
        rest="${tag%_s$seed}"
        gen="${rest%%_*}"
        tgt="${rest#${gen}_}"
        accept_one "$gen/$tgt/$seed" || rc=1
    done < <("$PY" "$HERE/manifest.py" --list --arm "$ARM")
else
    accept_one "$CELL" || rc=1
fi
exit $rc
