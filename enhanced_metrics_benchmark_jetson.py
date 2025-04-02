#!/usr/bin/env python3
import os
import pynvml

def is_jetson():
    """
    Checks if the system appears to be a Jetson device.
    It checks for the presence of '/etc/nv_tegra_release' and looks
    for 'jetson' in the '/proc/device-tree/model' file.
    """
    if os.path.exists("/etc/nv_tegra_release"):
        print("Found /etc/nv_tegra_release, assuming Jetson.")
        return True
    try:
        with open("/proc/device-tree/model", "r") as f:
            model = f.read().lower()
            if "jetson" in model:
                print("Found 'jetson' in /proc/device-tree/model, assuming Jetson.")
                return True
    except Exception as e:
        print("Error reading /proc/device-tree/model:", e)
    return False

async def linux_device_capabilities():
    """
    Asynchronously collects Linux device capabilities.
    It attempts to initialize NVML and query GPU memory info.
    If running on a Jetson or if any NVML error occurs, it bypasses these queries.
    """
    capabilities = {}
    
    # If we detect a Jetson device, bypass NVML completely.
    if is_jetson():
        print("Detected Jetson device – skipping NVML GPU memory queries.")
        capabilities["gpu_memory"] = "Not Supported on Jetson"
        return capabilities

    try:
        # Attempt NVML initialization and querying.
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        print(f"NVML initialized successfully. Device count: {device_count}")
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            try:
                gpu_memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                capabilities[f"gpu_{i}_memory"] = gpu_memory_info.total
            except pynvml.NVMLError_NotSupported as e:
                print(f"NVML query not supported for GPU {i}: {e}")
                capabilities[f"gpu_{i}_memory"] = "Not Supported"
            except Exception as err:
                print(f"Error querying GPU {i}: {err}")
                capabilities[f"gpu_{i}_memory"] = "Error"
    except Exception as e:
        # If any error occurs during NVML initialization or device queries,
        # we assume NVML is not supported and mark GPU memory as not supported.
        print("NVML initialization or query failed with error:", e)
        capabilities["gpu_memory"] = "Not Supported on Jetson"
    
    return capabilities

if __name__ == "__main__":
    import asyncio
    caps = asyncio.run(linux_device_capabilities())
    print("Device capabilities:", caps)
