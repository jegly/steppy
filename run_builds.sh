#!/usr/bin/env bash
cd "$(dirname "$0")"
log(){ echo "[$(date +%H:%M:%S)] $*"; }
for V in 16gb 32gb; do
  log "=== building $V ==="
  if ./build_deb.sh "$V" > "build-$V.log" 2>&1; then
    D="steppy_1.0.0-${V}_amd64.deb"
    log "$V built: $(( $(stat -c%s "$D") / 1048576 )) MiB — verifying"
    if dpkg-deb -c "$D" > "verify-$V.txt" 2> "verify-$V.err"; then
      log "$V VERIFIED — $(wc -l < verify-$V.txt) entries, $(grep -c 'fonts/licenses/' verify-$V.txt) licence files"
    else
      log "$V VERIFY FAILED: $(head -1 verify-$V.err)"
    fi
  else
    log "$V BUILD FAILED:"; tail -6 "build-$V.log"
  fi
done
log "ALL DONE"
ls -la steppy_1.0.0-*.deb 2>/dev/null | awk '{printf "  %.2f GiB  %s\n",$5/1073741824,$9}'
