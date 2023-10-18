# Project README

## Data

### Important Files

1. `/opt/Data/megadepth`: The data directory 
2. `configs/data/megadepth_trainval_**.py`: Configuration files directory for train and validation (Important parameters: `BASE_DIR`, `N_SAMPLES_PER_SUBSET`, `MGDPT_IMG_RESIZE`)
3. `configs/data/megadepth_test_1500.py`: Configuration files directory for testing 
   #### TODO: not implemented yet
4. `src/datasets/megadepth.py`: Dataset class for training; lines 84 - 85 change the input data for training to be DSM image as the second input

## Train

### Important Files

1. `configs/loftr/outdoor/loftr_ds_dense.py`: Configuration file for training
2. `train.py`: Train script
3. `scripts/reproduce_train/outdoor_ds.sh`: Train script for LoFTR
4. `MyLoFTR/logs/tb_logs`: The path to the logs directory is
5. Tensorboard: To run Tensorboard, use the following command: `tensorboard --logdir=logs --port=6006`, then open your browser and navigate to: [http://localhost:6006/](http://localhost:6006/)

## Docker

### Important Files

1. `docker/Dockerfile`: Docker file for training and inference
2. `docker/build_container.sh`: Script for building the Docker image (the script includes a command for running the container)
3. `docker/run_container.sh`: Script for running the Docker container 
#### Docker Commands:
1. Build the Docker image:
```
docker build -t *docker_image_name*:*tag_name* .
```

2. Run the Docker container:
```
docker run --gpus all -it --rm --shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864 -v *host_path*:*container_path* *docker_image_name*:*tag_name*
```
   - `--gpus all` for using all the GPUs
   - `-it` for interactive mode
   - `--rm` for removing the container after exiting
   - `--shm-size=8g` for setting the shared memory size to 8GB
   - `--ulimit memlock=-1 --ulimit stack=67108864` for setting the memory limit
   - `-v *host_path*:*container_path*` for mounting a directory from the host to the container
   - `*docker_image_name*:*tag_name*` for the Docker image name and tag

3. Save the Docker image to a tar file
```
docker save -o *file_name*.tar *docker_image_name*:*tag_name*
```
4. List all containers on your system, including both running and stopped containers
```
docker ps -a
``` 
5. Used to gracefully stop a running container
```
docker stop *container_name*
```
6. used to remove one or more containers from your system. The "-f" flag, also known as the force option, is used to forcefully remove a running container
```
docker rm -f *container_name*
```
7. List all images that are locally stored with the Docker Engine
```     
docker images
```
8. Used to remove one or more images from your system. The "-f" flag, also known as the force option, is used to forcefully remove an image
``` 
docker rmi -f *image_name*
``` 
9. Used to attach your terminal to a running container
```
docker attach *container_name*
```
10. Used to view the logs of a container
``` 
docker logs *container_name*
```
11. Used to view the processes running within a container
``` 
docker top *container_name*
```

## A100 Server (hpcts)

### Important Commands
1. Run the LoFTR Docker on 2 GPUs:
```
Runapp -c 2 -gpu2 -i loftr
``` 
   - `-gpu1` for 1 GPU, `-gpu4` for 4 GPUs, etc.
   - `-i` for the Docker name
   - `-c` for the number of CPUs
   #### TODO: need to ask Mark Hayat about the -c flag
2. Show the status and name of the running containers:
``` 
qstat
``` 
3. Delete the running container:
```
qdel "container_name"
``` 
4. Show the status of the GPUs (the MIG mode of the GPUs is shown at the end):
```
nvidia-smi
``` 
5. Open multiple terminals in the same window:
```
byobu
```
   - `F2` for a new terminal
   - `F3` and `F4` for switching between terminals
   - `F6` for closing a terminal

#### Our shared directory on hpcts server is 
```
/Projects/GS_AI_DEV
```

## Inference

### Important Files

1. `run_demo.py`: Inference script
2. `experiments/*`: All the experiments help files

## Evaluation

### Important Files
   
#### TODO: not implemented yet

## Visualization

### Important Files

1. `run_demo.py`: Visualization script
2. `experiments/*`: All the experiments help files

