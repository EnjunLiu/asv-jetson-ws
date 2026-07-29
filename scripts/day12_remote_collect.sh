#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 SLOT_ID LAYOUT_ID MOTION_STATE SCENE_SEED" >&2
  exit 2
fi

slot_id=$1
layout_id=$2
motion_state=$3
scene_seed=$4

[[ $slot_id =~ ^L[1-9]_S[01]_R[1-9][0-9]*$ ]] || {
  echo "DAY12_REMOTE_FAIL invalid slot_id=$slot_id" >&2
  exit 2
}
[[ $layout_id =~ ^L[1-9]$ ]] || {
  echo "DAY12_REMOTE_FAIL invalid layout_id=$layout_id" >&2
  exit 2
}
[[ $motion_state =~ ^S[01]$ ]] || {
  echo "DAY12_REMOTE_FAIL invalid motion_state=$motion_state" >&2
  exit 2
}
[[ $scene_seed =~ ^[0-9]+$ ]] || {
  echo "DAY12_REMOTE_FAIL invalid scene_seed=$scene_seed" >&2
  exit 2
}

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

set +u
source /opt/ros/humble/setup.bash
source install/setup.bash
set -u

automation_root="$repo_root/artifacts/day12_automation"
transfer_root="$repo_root/artifacts/pc_transfer"
mkdir -p "$automation_root" "$transfer_root"
launch_log="$automation_root/${slot_id}.launch.log"
: >"$launch_log"

launch_pid=
tail_pid=

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  if [[ -n ${launch_pid:-} ]] && kill -0 "$launch_pid" 2>/dev/null; then
    kill -INT -- "-$launch_pid" 2>/dev/null || true
    wait "$launch_pid" 2>/dev/null || true
  fi
  if [[ -n ${tail_pid:-} ]] && kill -0 "$tail_pid" 2>/dev/null; then
    kill "$tail_pid" 2>/dev/null || true
    wait "$tail_pid" 2>/dev/null || true
  fi
  exit "$exit_code"
}
trap cleanup EXIT INT TERM

if ss -ltnH 'sport = :8080' | grep -q .; then
  echo "DAY12_REMOTE_FAIL TCP port 8080 already has a listener" >&2
  exit 1
fi

echo "DAY12_REMOTE_START slot=$slot_id layout=$layout_id motion=$motion_state scene_seed=$scene_seed"

setsid env PYTHONUNBUFFERED=1 ros2 launch asv_bringup day12_collect.launch.py \
  slot_id:="$slot_id" \
  layout_id:="$layout_id" \
  motion_state:="$motion_state" \
  scene_seed:="$scene_seed" \
  >"$launch_log" 2>&1 &
launch_pid=$!

tail --pid="$launch_pid" -n +1 -F "$launch_log" &
tail_pid=$!

ready_deadline=$((SECONDS + 60))
while (( SECONDS < ready_deadline )); do
  if ! kill -0 "$launch_pid" 2>/dev/null; then
    echo "DAY12_REMOTE_FAIL ROS launch exited before readiness" >&2
    wait "$launch_pid" || true
    exit 1
  fi
  if grep -q "DAY8_RECORDER_READY" "$launch_log" \
    && ss -ltnH 'sport = :8080' | grep -q .; then
    echo "DAY12_REMOTE_READY slot=$slot_id layout=$layout_id motion=$motion_state scene_seed=$scene_seed"
    break
  fi
  sleep 0.2
done

if ! grep -q "DAY8_RECORDER_READY" "$launch_log" \
  || ! ss -ltnH 'sport = :8080' | grep -q .; then
  echo "DAY12_REMOTE_FAIL timed out waiting for recorder and TCP listener" >&2
  exit 1
fi

completion_seen=false
while kill -0 "$launch_pid" 2>/dev/null; do
  if grep -q "DAY8_RECORDING_COMPLETE episode=" "$launch_log"; then
    completion_seen=true
    break
  fi
  sleep 0.2
done

if [[ $completion_seen == true ]] && kill -0 "$launch_pid" 2>/dev/null; then
  recorder_pid=$(
    pgrep -P "$launch_pid" -f "record_episode" | head -n 1 || true
  )
  if [[ -n $recorder_pid ]]; then
    kill -TERM "$recorder_pid" 2>/dev/null || true
  fi

  shutdown_deadline=$((SECONDS + 10))
  while kill -0 "$launch_pid" 2>/dev/null \
    && (( SECONDS < shutdown_deadline )); do
    sleep 0.2
  done
  if kill -0 "$launch_pid" 2>/dev/null; then
    kill -TERM -- "-$launch_pid" 2>/dev/null || true
  fi
fi

set +e
wait "$launch_pid"
launch_status=$?
set -e
launch_pid=

if [[ $completion_seen != true ]]; then
  echo "DAY12_REMOTE_FAIL ROS launch exited before recording completed status=$launch_status" >&2
  exit 1
fi

if [[ -n ${tail_pid:-} ]]; then
  wait "$tail_pid" 2>/dev/null || true
  tail_pid=
fi

episode_dir=$(
  grep "DAY8_RECORDING_COMPLETE episode=" "$launch_log" \
    | tail -n 1 \
    | sed -E 's/.*episode=([^ ]+) frames=.*/\1/'
)
if [[ -z $episode_dir || ! -d $episode_dir ]]; then
  echo "DAY12_REMOTE_FAIL no completed episode was found in $launch_log" >&2
  exit 1
fi

run_id=$(basename "$episode_dir")
supervised_dir="$repo_root/artifacts/day10_supervised/$run_id"

ros2 run asv_vla evaluate_episode "$episode_dir" --min-frames 80
ros2 run asv_vla build_supervised_dataset \
  --episode "$episode_dir" \
  --instructions dataset/language/instructions.jsonl \
  --output "$supervised_dir"
ros2 run asv_vla evaluate_supervised_dataset \
  "$supervised_dir" \
  --require-all-labels

python3 -m training.day12_collection status \
  --data-root . \
  --report artifacts/day12_collection_report_v1.json
python3 -m training.dataset_registry \
  --data-root . \
  --output artifacts/day12_registry/dataset_registry_v1.jsonl
python3 -m training.make_group_splits \
  --registry artifacts/day12_registry/dataset_registry_v1.jsonl \
  --output artifacts/day12_registry/group_split_v1.json \
  --instructions dataset/language/instructions.jsonl

package_path="$transfer_root/day12_${slot_id}_${run_id}.tar.gz"
tar -czf "$package_path" \
  dataset/language/instructions.jsonl \
  "artifacts/day8_episode/$run_id" \
  "artifacts/day10_supervised/$run_id"
package_sha=$(sha256sum "$package_path" | awk '{print $1}')

echo "DAY12_PACKAGE path=$package_path sha256=$package_sha"
echo "DAY12_REMOTE_COMPLETE slot=$slot_id run_id=$run_id"
