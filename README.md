![Pylint](https://github.com/eencloud/uv-vapipeline-classifier-v2/actions/workflows/pylint.yml/badge.svg)

# uv-vapipeline-classifier-v2

## Onnx models
- Models(onnx) should be stored in folder.
  
  `cd ./onnx_models`
- In `.gitattributes` file add the following line to add a onnx file stored at filepath

  `filepath filter=lfs diff=lfs merge=lfs -text`

<details open="open">
<summary><h4 style="display: inline-block">Git LFS tutorial</h4></summary>
  <uol>
    <li>To set up git lfs for your account (needs to be done only once)</li>
    git lfs install
    <li>Track models by git lfs</li>
    git lfs track "*.h5"
    <li>Make sure .gitattributes is tracked</li>
     git add .gitattributes
  </uol>
</details>

## Local Testing  
- `sudo docker run --rm -v $(pwd):/app -it uvdeployment/shield:uv-pipeline-cudakernel bash`
- `cd /app/src`
- `mkdir -p build`
- `cd build`
- `cmake ..`
- `make`
- `cd ..`
- 'python3 test_the_trt.py'

## Production Build
- `sudo make buildd`

## Production Test
- `cd test`
- Follow steps in the README.md

## Description

- For adding cuda kernels, make changes to src/CMakeLists.txt
- process_utils corresponds to pipeline process
- ctype_utils handles the Python to C calling arguments
- test_the_trt.py is for local testing
- preprocess depends on Module requirements