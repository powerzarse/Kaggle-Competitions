#!/usr/bin/env bash
# Pusht ein Kaggle-Notebook (Enveda), wartet auf den Commit-Lauf und gibt es dann ab. Benötigt tools/kg als "kg" im PATH.
# Pusht ein Kaggle-Notebook, sobald ein GPU-Platz frei ist, wartet auf den Commit-Lauf und gibt dann ab.
# Aufruf: push_and_submit.sh <notebook-ordner> <owner/slug> <nachricht>
set -u
DIR="$1"; REF="$2"; MSG="$3"; C=enveda-CASMI26-molecule-id-mass-spectra
log(){ echo "$(date -u +%H:%M:%S) $*"; }
for i in $(seq 1 360); do
  out=$(kg kernels push -p "$DIR" 2>&1)
  if echo "$out" | grep -q "Maximum batch GPU session"; then sleep 120; continue; fi
  log "push: $(echo "$out" | tail -2 | tr '\n' ' ')"; break
done
echo "$out" | grep -qi "error" && { log "push failed"; exit 1; }
VER=$(echo "$out" | grep -o 'version [0-9]*' | grep -o '[0-9]*' | tail -1)
log "version=${VER:-unknown}"
for i in $(seq 1 240); do
  st=$(kg kernels status "$REF" 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z_]*')
  case "$st" in
    *COMPLETE) log "commit run complete"; break;;
    *ERROR|*CANCEL*) log "commit run $st"; exit 1;;
  esac
  sleep 60
done
case "$st" in *COMPLETE) ;; *) log "commit run not complete after waiting ($st), no submission"; exit 1;; esac
VARGS=(); [ -n "${VER:-}" ] && VARGS=(-v "$VER")
log "submit: $(kg competitions submit -c $C -f submission.csv -k "$REF" "${VARGS[@]}" -m "$MSG" 2>&1 | tail -2 | tr '\n' ' ')"
