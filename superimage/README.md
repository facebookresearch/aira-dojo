# 🪐 Superimage 🪐

Superimage is a container image for Docker and Apptainer that simulates a virtual machine. It integrates well with job schedulers like Slurm, requires no root permissions.

Superimage is primarily intended for the AI Research Agents project, serving as the environment to run agent actions.  It is designed to be simple but not necessarily minimal, including most of the software, packages, and tools needed for easy use by agents.

## Building

To build the image:
```
NVIDIA_VISIBLE_DEVICES=0 apptainer build --nv --nvccli --fakeroot path/to/superimage.root.sif apptainer.def
```