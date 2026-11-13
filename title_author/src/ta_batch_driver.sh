#!/bin/bash
# Runs a split through the Gemini Batch API in chunks, one chunk at a time, because the project only allows a limited
# number of tokens in active batch jobs at once (a whole 2,384-work test set is rejected with HTTP 429; 600 works fit).
# The key comes only from the environment, never from this file.
#
#   GEMINI_API_KEY=... TA_PQ=<parquet folder> ALLOW_TEST=1 THINK_LEVEL=low \
#   [NSAMP=1400] title_author/src/ta_batch_driver.sh PROMPT.md SPLIT MODEL TOTAL [CHUNK]
#   NSAMP: use a random sample of that many works (fixed seed 0) instead of the whole split; TOTAL is how many of them to run.
#
#   e.g. title_author/src/ta_batch_driver.sh title_author/data/gemini_prompts/archive/gepa_v2.md test gemini-3.8-flash 2384 600
#
# For each chunk: submit, wait until the job is done, collect and score it. When all chunks are done, merge them with
#   python title_author/src/ta_gemini_batch.py combine NAME TAG1 TAG2 ...
set -u
cd "$(dirname "$0")/../.." || exit 1
PROMPT=$1; SPLIT=$2; MODEL=$3; TOTAL=$4; CHUNK=${5:-600}; NSAMP=${NSAMP:-0}
LBL=$([ "$NSAMP" = "0" ] && echo all || echo "$NSAMP")
B="python3 title_author/src/ta_gemini_batch.py"
STEM=$(basename "$PROMPT" .md)
TH=""; [ -n "${THINK_LEVEL:-}" ] && TH="_think-$THINK_LEVEL"
for ((A = 0; A < TOTAL; A += CHUNK)); do
  E=$((A + CHUNK)); R="$A:$E"
  T="batch_${STEM}_${MODEL}_${SPLIT}_${LBL}${TH}_rows${A}-${E}"
  if [ ! -f "title_author/data/gemini_batches/$T.json" ]; then
    until ROWS=$R $B submit "$PROMPT" "$SPLIT" "$MODEL" "$NSAMP" 2>&1 | grep -q "^TAG"; do
      echo "$(date +%T) submit $R not accepted yet (token cap or error), waiting"; sleep 60
    done
  fi
  while true; do
    out=$($B status "$T" 2>&1 | grep -v -i warn | tail -1); echo "$(date +%T) $out"
    case "$out" in
      *SUCCEEDED*) $B collect "$T" 2>&1 | grep -v -i warn | tail -3 | cut -c1-160; break;;
      *FAILED*|*CANCELLED*|*EXPIRED*) echo "chunk $R ended badly: $out"; exit 1;;
    esac
    sleep 30
  done
done
echo ALL_CHUNKS_DONE
