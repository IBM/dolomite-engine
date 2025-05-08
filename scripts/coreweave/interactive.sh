CONTAINER=/mnt/vast/squash/ete-dolomite.sqsh
MOUNT=/mnt:/mnt

srun -N1  --container-image=$CONTAINER  --container-mounts=$MOUNT --no-container-remap-root --pty bash
