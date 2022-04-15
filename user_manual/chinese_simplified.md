[[_TOC_]]

---
# 简介
OpenVINO 是 Intel 推出的基于 Intel 设备 (CPU, GPU including integrated graphics, Movidius VPU, FPGA among others) 的类神经网络 **Inference** 部署工具，支持 Windows, Linux 等操作系统

## 优势
加速模型 Inference 只需有 Intel CPU、集成显卡，或 Movidius 神经计算棒与即将面世的 Intel 独立显卡等。不需NVIDIA 显卡，即可部署模型至算力较差的边缘设备

测试平台: Intel i5 10500, Deeplab v3 plus with resnest101, 单位: 秒

只使用 ONNX-CPU 耗时:
```
infer finished, avg time is 2.0437190333409094
```
使用 Open VINO 耗时:
```
infer finished, avg time is 0.3548231950446741
```
速度提升约 6 倍

## 使用流程

1. 在原开发环境中训练一个模型，转换成 Intel OpenVINO 支持的模型格式 （通常为ONNX）
2. 使用 Intel OpenVINO Model Optimizer 编译模型并优化，生成 OpenVINO Intermediate Representation (IR) 的中介文件：包含网络结构 `.xml` 与网络权重 `.bin`
3. 调用 Intel OpenVINO Inference Engine 进行网络计算

## 下载地址 
https://www.intel.cn/content/www/cn/zh/developer/tools/openvino-toolkit-download.html

# 安装 & 配置 & 测试
## Windows部署
### 安装
运行安装文件，如提示缺依赖项，可忽略并在安装后按需配置

安装后，以 OpenVINO_2021.4.752(LTS) 为例，其安装路径位于
```
C:\Program Files (x86)\Intel\openvino_2021.4.752\
```
### 配置环境变量

请设置**系统**环境变量

参考网站如下，请留意 `INTEL_OPENVINO_DIR	` 的实际版本
https://www.intel.com/content/www/us/en/support/articles/000033440/software/development-software.html

### 配置 python 环境
使用 pip 安装 conda-python 的 OpenVINO libraries
```
pip install openvino
```

### 配置优化器

若要安装所有优化器，请运行以下指令。也可单独运行个别优化器的安装脚本
```
cd "C:\Program Files (x86)\Intel\openvino_2021.4.752\deployment_tools\model_optimizer\install_prerequisites"
.\install_prerequisites.bat
```
### 测试

运行 demo 文件需在系统中部署好 Visual Studio Build Tools（Visual Studio 生成工具）。可下载  Visual Studio Installer 利用该工具安装

Visual Studio Installer 下载地址: https://visualstudio.microsoft.com/zh-hans/downloads/

#### 性能测试

因国内存在针对 `raw.githubusercontent.com` 的域名解析污染，因此需设置 `hosts` 以下载性能测试的所需文件

请访问 https://ipaddress.com/website/raw.githubusercontent.com 以查询该网站的 IP 地址，再修改 `hosts` 于文件尾端添加以下内容 （2021/12/30验证通过）
```
185.199.109.133 raw.githubusercontent.com
```

修改 `C:\Program Files (x86)\Intel\openvino_2021.4.752\deployment_tools\demo\demo_benchmark_app.bat` 中的 
`Line 216`为以下内容（适用Visual Studio Build Tools 2022）
```
cd /d "%INTEL_OPENVINO_DIR%\deployment_tools\inference_engine\samples\cpp" && cmake -E make_directory "%SOLUTION_DIR64%" && cd /d "%SOLUTION_DIR64%" && cmake -G "Visual Studio 17 2022" -A %PLATFORM% "%INTEL_OPENVINO_DIR%\deployment_tools\inference_engine\samples\cpp"
```

运行以下指令，脚本会自动下载 demo 权重并利用 Visual Studio Build Tools 编译

```
cd "C:\Program Files (x86)\Intel\openvino_2021.4.752\deployment_tools\demo\"
.\demo_benchmark_app.bat
```

运行结束时，程序会输出 demo 测试得到的速度信息
![benchdemo](https://user-images.githubusercontent.com/79516102/163554620-6f20ea47-46f5-4af3-8020-b12cf9bc4dc1.PNG)

#### 汽车辨识测试

如法炮制，修改脚本 `C:\Program Files (x86)\Intel\openvino_2021.4.752\deployment_tools\demo\demo_security_barrier_camera.bat` 
在 `Line 216` 替换 `"Visual Studio !MSBUILD_VERSION!"` 为 `"Visual Studio 17 2022"` 以指定编译构建工具
在 `Line 95`, `Line 96`的头部添加 `rem `（其后间隔一空格）注解下载程序，跳过下载步骤

OpenVINO提供的 demo 模型**下载后储存的位置**如下
```
%USERPROFILE%\Documents\Intel\OpenVINO\openvino_models\ir\intel
```

运行测试需要三组模型，分别是
```
license-plate-recognition-barrier-0001
vehicle-attributes-recognition-barrier-0039
vehicle-license-plate-detection-barrier-0106
```

OpenVINO 的默认模型的下载地址记录在以下文件夹中
```
C:\Program Files (x86)\Intel\openvino_2021.4.752\deployment_tools\open_model_zoo\models\intel\
```
因其中默认的下载源 https://storage.openvinotoolkit.org 在国内无法访问，**请手动进入如下网站下载**并保存至上述位置对应的文件夹内。因 demo 只测试 Float 16 精度的网络，因此下载 `FP16` 文件夹即可。

license-plate-recognition-barrier-0001:
https://download.01.org/opencv/2021/openvinotoolkit/2021.1/open_model_zoo/models_bin/2/license-plate-recognition-barrier-0001/

vehicle-attributes-recognition-barrier-0039:
https://download.01.org/opencv/2021/openvinotoolkit/2021.1/open_model_zoo/models_bin/2/vehicle-attributes-recognition-barrier-0039/

vehicle-license-plate-detection-barrier-0106:
https://download.01.org/opencv/2021/openvinotoolkit/2021.1/open_model_zoo/models_bin/2/vehicle-license-plate-detection-barrier-0106/

其中记录网络结构的 `.xml` 之 `Line 2` 中的 `version` 记录了模型结构和对应中介文件 `.bin` 的 IR 版本。如果出现类似以下提示，请检查该行信息，并用新版的模型 IR 文件进行 Inference。建议至少使用 `version="10"` 的中介模型
```
The support of IR v7 has been removed from the product. Please, convert the original model using the Model Optimizer which comes with this version of the OpenVINO to generate supported I
R version.
```

运行以下指令，脚本会编译并运行汽车辨识测试

```
cd "C:\Program Files (x86)\Intel\openvino_2021.4.752\deployment_tools\demo\"
.\demo_security_barrier_camera
```
如一切正常，会显示以下运行结果
![Detection_result](https://user-images.githubusercontent.com/79516102/163554744-2b6132cf-54ab-4791-a95a-2d4e96cdbda6.PNG)
