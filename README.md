### Ascend910 OP Test Samples

## How to Run

Pre-requisites: a server with Ascend910, refer to https://www.hiascend.com/software/cann/community

1. Get runtime docker image

  > Note: the installed CANN version in this image is '5.0.2.alpha005'.

  ```bash
  # x86_64 host
  docker pull qili93/develop:latest-dev-cann5.0.2.alpha005-gcc82-x86_64

  # aarch64 host
  docker pull qili93/develop:latest-dev-cann5.0.2.alpha005-gcc82-aarch64
  ```

2. Start docker container

  ```bash
  docker run -it --name qili93-dev-npu -v /home/<username>:/workspace \
             --pids-limit 409600 --network=host --shm-size=128G \
             --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
             --device=/dev/davinci2 \ # Note: change this if need to mapping other device ID
             --device=/dev/davinci_manager \
             --device=/dev/devmm_svm \
             --device=/dev/hisi_hdc \
             -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
             -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
             -v /usr/local/dcmi:/usr/local/dcmi \
             qili93/develop:latest-dev-cann5.0.2.alpha005-gcc82-x86_64 /bin/bash
  ```

3. Compile and run by `run_demo.sh`, for example:

  ```bash
  # Run Add OP
  sh run_demo.sh Add

  # Run ResizeNearestNeighborV2 OP
  sh run_demo.sh ResizeNearestNeighborV2
  ```

4. Tracking issues here (i.e. issue links to Ascend community)

  - https://gitee.com/ascend/modelzoo/issues/I44MV8?from=project-issue # Resize
  - https://gitee.com/ascend/modelzoo/issues/I44RAL?from=project-issue # ResizeD
  - https://gitee.com/ascend/modelzoo/issues/I47UIG?from=project-issue # npu_deformable_conv2d

