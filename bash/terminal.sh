#!/bin/bash

# Interactive prep for work on the compute node: prompts tmux -> jupyter -> node -> kill,
# then cleanup, optional Jupyter tunnel from this host ("kill port" applies only to Jupyter).

# Shared HPC layout (NFS): notebooks / workload under scratch; home is typically /home/$USER.
HPC_SCRATCH="${HPC_SCRATCH:-/scratch/aa9360/supercloud_power}"

# Jupyter prefers this port; if it is busy on the login node, the next free port is used instead.
JUPYTER_PREFERRED_PORT="${JUPYTER_PORT:-8891}"

yn_prompt() {
    local msg="$1"
    local reply
    while true; do
        read -r -p "${msg} (y/N/q quit): " reply
        case "${reply,,}" in
            q | quit) echo "Exiting."; exit 0 ;;
            y | yes) return 0 ;;
            n | no | "") return 1 ;;
            *) echo "Invalid input. Use y, n, or q." ;;
        esac
    done
}

# First free local TCP port in [start, start+span) (for SSH -L bind). Uses fuser.
first_free_local_port() {
    local start="$1"
    local span="${2:-100}"
    local p end=$((start + span))
    for ((p = start; p < end; p++)); do
        if ! fuser "${p}/tcp" &>/dev/null; then
            echo "${p}"
            return 0
        fi
    done
    return 1
}

# Uses globals: node, HPC_SCRATCH, JUPYTER_LISTEN_PORT (set in Jupyter block)
open_compute_shell() {
    local qs
    qs=$(printf '%q' "$HPC_SCRATCH")
    local tmsg=""
    [[ -n "${JUPYTER_LISTEN_PORT:-}" ]] && tmsg=" Tunnel may still use local port ${JUPYTER_LISTEN_PORT}."
    echo >&2 "Opening interactive shell on ${node} (scratch or \$HOME).${tmsg}"
    # shellcheck disable=SC2029
    ssh -t "$node" "cd ${qs} 2>/dev/null || cd \"\${HOME}\" || true; exec bash -l"
}

# --- 1) tmux on compute -------------------------------------------------------
echo "=== tmux ==="
if yn_prompt "Use tmux on the compute node"; then
    session_cmd="tmux"
    use_tmux=1
else
    session_cmd=""
    use_tmux=0
fi

# --- 2) Jupyter + tunnel from this host ---------------------------------------
echo "=== Jupyter ==="
if yn_prompt "Start Jupyter and an SSH tunnel from this host (prefers port ${JUPYTER_PREFERRED_PORT}, or next free if busy)"; then
    use_jupyter=1
else
    use_jupyter=0
fi

# --- 3) compute node -----------------------------------------------------------
echo "=== node ==="
nodes=$(squeue -u "$USER" 2>/dev/null | awk 'NR>1 {print $8}' | sort -u | grep -v '^$' || true)
if [[ -z "$nodes" ]]; then
    node_count=0
else
    node_count=$(echo "$nodes" | grep -c . || true)
fi

if [[ "$node_count" -eq 0 ]]; then
    echo "No nodes found in your squeue (or squeue failed)."
    read -r -p "Compute hostname to ssh into (e.g. cs743): " node
elif [[ "$node_count" -eq 1 ]]; then
    node=$(echo "$nodes" | head -n 1)
    echo "Only one node in squeue: $node"
    read -r -p "Press Enter to use it, or type a different hostname: " node_override
    if [[ -n "${node_override:-}" ]]; then
        node="$node_override"
    fi
else
    echo "Available NODELISTs from squeue:"
    echo "$nodes"
    read -r -p "Which host do you want to ssh into? " node
fi

if [[ -z "${node:-}" ]]; then
    echo "No node selected. Exiting."
    exit 1
fi

# --- 4) Jupyter: optional kill on *preferred* port only (no port prompts if not Jupyter) ---
do_kill=0
if [[ "$use_jupyter" -eq 1 ]]; then
    echo "=== kill (Jupyter preferred port ${JUPYTER_PREFERRED_PORT} only) ==="
    kill_msg="Kill processes on TCP port ${JUPYTER_PREFERRED_PORT} on this host and on ${node}? (Jupyter tries this port first; if it stays busy, the next free port is used)"
    if yn_prompt "$kill_msg"; then
        do_kill=1
    fi
fi

# --- execute ------------------------------------------------------------------
if [[ "$do_kill" -eq 1 ]]; then
    echo "Clearing ${JUPYTER_PREFERRED_PORT}/tcp on $(hostname -s 2>/dev/null || hostname) ..."
    fuser -k "${JUPYTER_PREFERRED_PORT}/tcp" 2>/dev/null || true
    echo "Clearing ${JUPYTER_PREFERRED_PORT}/tcp on ${node} ..."
    ssh "$node" "fuser -k ${JUPYTER_PREFERRED_PORT}/tcp 2>/dev/null || true" || echo "(Could not run fuser on ${node}; continuing.)"
fi

if [[ "$use_jupyter" -eq 1 ]]; then
    if ! JUPYTER_LISTEN_PORT="$(first_free_local_port "${JUPYTER_PREFERRED_PORT}")"; then
        echo >&2 "No free TCP port on this host from ${JUPYTER_PREFERRED_PORT} upward (tried 100 ports). Exiting."
        exit 1
    fi
    echo "Jupyter will listen on port ${JUPYTER_LISTEN_PORT} (tunnel: localhost:${JUPYTER_LISTEN_PORT} -> ${node}:${JUPYTER_LISTEN_PORT})."

    TMUX_SESSION="${TMUX_SESSION:-supercloud}"
    if [[ -n "${JUPYTER_REMOTE_CD:-}" ]]; then
        NOTEBOOK_DIR="$JUPYTER_REMOTE_CD"
    else
        NOTEBOOK_DIR="${JUPYTER_NOTEBOOK_DIR:-$HPC_SCRATCH}"
    fi

    quoted_dir=$(printf '%q' "$NOTEBOOK_DIR")
    # jpy_cmd="set -e; d=${quoted_dir}; if [[ ! -d \"\$d\" ]]; then mkdir -p \"\$d\" || { echo \"Cannot use notebook dir: \$d\" >&2; exit 1; }; fi; cd \"\$d\" && fuser -k ${JUPYTER_LISTEN_PORT}/tcp 2>/dev/null || true; exec jupyter notebook --no-browser --port=${JUPYTER_LISTEN_PORT}"
    jpy_cmd="set -e; d=${quoted_dir}; if [[ ! -d \"\$d\" ]]; then mkdir -p \"\$d\" || { echo \"Cannot use notebook dir: \$d\" >&2; exit 1; }; fi; cd \"\$d\" && fuser -k ${JUPYTER_LISTEN_PORT}/tcp 2>/dev/null || true; exec jupyter lab --no-browser --port=${JUPYTER_LISTEN_PORT}"

    if [[ "$do_kill" -ne 1 ]]; then
        fuser -k "${JUPYTER_LISTEN_PORT}/tcp" 2>/dev/null || true
    fi

    if ! ssh -f -N -o ExitOnForwardFailure=yes -L "${JUPYTER_LISTEN_PORT}:127.0.0.1:${JUPYTER_LISTEN_PORT}" "$node"; then
        echo >&2 "SSH tunnel to ${node}:${JUPYTER_LISTEN_PORT} failed. Try: fuser -k ${JUPYTER_LISTEN_PORT}/tcp"
        open_compute_shell
        exit 1
    fi
    echo "Jupyter: connect from this host using http://localhost:${JUPYTER_LISTEN_PORT}/... (token appears in the Jupyter log on ${node})."

    run_jupyter_ssh() {
        if [[ "$use_tmux" -eq 1 ]]; then
            # shellcheck disable=SC2029
            ssh -t "$node" "exec tmux new-session -A -s ${TMUX_SESSION} bash -lc $(printf '%q' "$jpy_cmd")"
        else
            # shellcheck disable=SC2029
            ssh -t "$node" "bash -lc $(printf '%q' "$jpy_cmd")"
        fi
    }

    if ! run_jupyter_ssh; then
        echo >&2 "Jupyter session ended non-zero. Opening a normal shell on ${node}."
        open_compute_shell
    fi
    exit 0
fi

qscratch=$(printf '%q' "$HPC_SCRATCH")
if [[ -n "$session_cmd" ]]; then
    # shellcheck disable=SC2029
    ssh -t "$node" "cd ${qscratch} && bash -l -c $(printf '%q' "$session_cmd")"
else
    # shellcheck disable=SC2029
    ssh -t "$node" "cd ${qscratch} && exec bash -l"
fi
