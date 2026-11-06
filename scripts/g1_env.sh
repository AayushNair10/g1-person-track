# Laptop-side settings for talking to the robot. Source it:  source scripts/g1_env.sh
# Uses one shared SSH connection (you type the password once):
#   ssh -fNM -o ControlPath=~/.ssh/cm-g1 -o ControlPersist=12h unitree@192.168.123.164
G1_HOST=${G1_HOST:-unitree@192.168.123.164}
G1_DIR=${G1_DIR:-aayush_person_track}          # relative to the robot's home: /home/unitree/aayush_person_track
G1_CTL=${G1_CTL:-$HOME/.ssh/cm-g1}
G1_SSH_OPTS=(-o ControlPath="$G1_CTL" -o ConnectTimeout=5)

g1_check() {  # is the shared connection up?
    if ! ssh "${G1_SSH_OPTS[@]}" -O check "$G1_HOST" 2>/dev/null; then
        echo "No shared SSH connection. In your own terminal run:"
        echo "  ssh -fNM -o ControlPath=$G1_CTL -o ControlPersist=12h $G1_HOST"
        return 1
    fi
}
# Run a command on the robot (stdin closed so the ROS prompt in .bashrc cannot block it).
g1_ssh() { ssh -n "${G1_SSH_OPTS[@]}" "$G1_HOST" "$@"; }
