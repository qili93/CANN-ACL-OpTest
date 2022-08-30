### Ascend910 OP Test Samples

## How to Run

Pre-requisites: a server with Ascend910, refer to https://www.hiascend.com/software/cann/community

1. Get runtime docker image

  > Note: the installed CANN version in this image is '5.0.2.alpha005'.

  ```bash
  # x86_64 host
  docker pull registry.baidubce.com/device/paddle-npu:cann512-x86_64-gcc75

  # aarch64 host
  docker pull registry.baidubce.com/device/paddle-npu:cann512-aarch64-gcc75
  ```

2. Start docker container

  ```bash
  docker run -it --name cann512 -v `pwd`:/workspace --workdir=/workspace \
        --pids-limit 409600 -v /home/datasets:/datasets \
        --privileged --network=host --shm-size=128G \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/dcmi:/usr/local/dcmi \
        registry.baidubce.com/device/paddle-npu:cann512-x86_64-gcc82 /bin/bash
  ```

3. Compile and run by `run_demo.sh`, for example:

  ```bash
  # Run FillV2 OP
  sh run_demo.sh FillV2

  # Run ResizeNearestNeighborV2 OP
  sh run_demo.sh ResizeNearestNeighborV2
  ```

4. Tracking issues here (i.e. issue links to Ascend community)

  - https://gitee.com/ascend/modelzoo/issues/I44MV8?from=project-issue # Resize
  - https://gitee.com/ascend/modelzoo/issues/I47UIG?from=project-issue # npu_deformable_conv2d
  - https://gitee.com/ascend/modelzoo/issues/I4ENI7?from=project-issue # Sort output mismatch
  - https://gitee.com/ascend/modelzoo/issues/I4EO3N?from=project-issue # Sort error in INT64
  - https://gitee.com/ascend/modelzoo/issues/I4DKLV?from=project-issue # Fill error on CANN 5.0.3

