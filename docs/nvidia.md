## Installing NVIDIA CUDA drivers & toolkit (rbgquanta)
NVIDIA Driver and CUDA toolkit installation: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

### Pre-Installation: 
1. Verify You Have a CUDA-Capable GPU
    * `sudo update-pciids`
    * `sudo lspci -v | grep NVIDIA`
    * check that the graphics card is listed under one of the https://developer.nvidia.com/cuda-gpus categories

2. Verify You Have a Supported Version of Linux
    * `lsb_release -a`, or 
    * `uname -m && cat /etc/*release`
    * the first line (x86_64) gives the bit-system, and DISTRIB_RELEASE gives the release version 

3. Verify the system has gcc installed
    * `gcc --version`, o.w.
    * `sudo apt-get install build-essential`

4. Verify the System has the Correct Kernel Headers and Development Packages Installed
    * `uname -r` (4.15.0-99-generic)
    * This is the version of the kernel headers and development packages that must be installed prior to installing the CUDA Drivers.
    * Check if kernel headers are already installed and match kernel version:
    * `ls -l /usr/src/linux-headers-$(uname -r)`
    * o.w., install:
    * `sudo apt-get install linux-headers-$(uname -r)`

5. Choose an Installation Method
    * visit: http://developer.nvidia.com/cuda-downloads
    * Linux > x86_64 > Ubuntu > 18.04 > deb (local)


### Package Manager Installation:
```shell
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
```
- Compare to checksum file (second to last line): https://developer.download.nvidia.com/compute/cuda/10.2/Prod/docs/sidebar/md5sum.txt
- md5sum cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb

### Post-installation Actions:
1. Environment Setup
    * find Nsight Compute version: `nv-nsight-cu-cli -v`

    > Note:
    > By default, when installing from a Linux .run file, NVIDIA Nsight Compute is located in /usr/local/cuda-[cuda-version]/nsight-compute-[version].
    >When installing from a .deb or .rpm package, it is located in /opt/nvidia/nsight-compute/[version] 
    * `export PATH=/usr/local/cuda-10.2/bin:/opt/nvidia/nsight-compute/2019.5.0${PATH:+:${PATH}}`

2.  Install Persistence Daemon (as root)
    * `sudo /usr/bin/nvidia-persistenced --verbose`

3. Install Writable (Example) Samples
    * `cuda-install-samples-10.2.sh ~/NVIDIA_CUDA-10.2_Samples`
    
4. Verify the Installation
    * Verify the Driver Version: `cat /proc/driver/nvidia/version`
    * Compiling the Examples: `nvcc -V`
    * Compiling the Examples: `cd ~/NVIDIA_CUDA-10.2_Samples/NVIDIA_CUDA-10.2_Samples/ `
    * type: `make -k`
    * To run the binaries type: `deviceQuery`
    * compare output with https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#install-samples
    
    > Note: Expected failure to compile: https://forums.developer.nvidia.com/t/where-is-nvscibuf-h/107802/19