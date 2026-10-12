# The runtime's image: Python, and the Vulkan userspace an AMD card's engines run on. Pin the Mesa
# packages to the host's exact versions (mesa_check warns when they differ: RADV sets numerics).
# NVIDIA cards need nothing here: the NVIDIA container toolkit mounts the driver and nvidia-smi.
FROM ubuntu:24.04
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 mesa-vulkan-drivers libvulkan1 \
 && rm -rf /var/lib/apt/lists/*
