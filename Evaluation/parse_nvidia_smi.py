# source https://gist.github.com/EndruK/c2c22a9161b050c714dd9d8788bd420f
import subprocess


def get_nvidia_smi_string(id=-1):
    """
    Get nvidia-smi result either for the system or for a specific GPU
    :param id: PCI GPU id (default=1 for system)
    :return: nvidia-smi string
    """
    if id == -1:
        nvidia_smi = subprocess.run(["nvidia-smi", "-q"], stdout=subprocess.PIPE)
    else:
        nvidia_smi = subprocess.run(["nvidia-smi", "-i", str(id), "-q"], stdout=subprocess.PIPE)
    result = nvidia_smi.stdout.decode()
    return result


def get_num_of_gpus():
    """
    Get the number of GPUs on the system
    :return: number of GPUs
    """
    num = 0
    smi = get_nvidia_smi_string()
    for line in smi.split("\n"):
        if "Attached GPUs" in line:
            num = int(line.split(": ")[1])
            break
    return num


def get_gpu_memory_single_gpu(gpu_id):
    """
    Get the memory capacity for a single GPU
    :param gpu_id: the PCI id of a GPU
    :return: tuple(int, String) holding the value and unit of the given GPU mem
    """
    memory_usage_target = "    FB Memory Usage"
    lines = get_nvidia_smi_string(gpu_id).split("\n")
    mem = ""
    for i in range(len(lines)):
        if lines[i].startswith(memory_usage_target):
            end = i
            for j in range(i, len(lines)):
                if lines[j].startswith("    BAR1 Memory Usage"):
                    end = j - 1
                    break
            mem = lines[i:end]
            break
    size = [line.split(": ")[1] for line in mem if line.startswith("        Total")][0]
    size_elements = size.split(" ")
    size = int(size_elements[0])
    unit = size_elements[1]
    return size, unit


def get_num_processes_on_gpu(gpu_id):
    """
    Get the number of processes running on a GPU
    :param gpu_id: the PCI id of a GPU
    :return: number of processes
    """
    smi = get_nvidia_smi_string(gpu_id)
    number_processes = smi.count("Process ID")
    return number_processes


def get_processes_on_gpu(gpu_id):
    """
    Get all processes running on the given GPU
    :param gpu_id: the PCI id of a GPU
    :return: a list of all processes of the GPU
    """
    smi = get_nvidia_smi_string(gpu_id)
    processes = []
    process_lines = []
    lines = smi.split("\n")
    for i in range(len(lines)):
        if lines[i].startswith("    Processes"):
            processes_lines = lines[i + 1:]
    process_element = {}
    for process_line in processes_lines:
        if process_line.startswith("        Process ID"):
            if len(process_element) > 0:
                processes.append(process_element)
                process_element = {}
            process_element["process_id"] = int(process_line.split(": ")[1])
        if process_line.startswith("            Type"):
            process_element["Type"] = process_line.split(": ")[1]
        if process_line.startswith("            Name"):
            process_element["Name"] = process_line.split(": ")[1]
        if process_line.startswith("            Used GPU Memory"):
            string = process_line.split(": ")[1]
            tmp = string.split(" ")
            mem = tmp[0]
            unit = tmp[1]
            process_element["Memory"] = int(mem)
            process_element["Unit"] = unit
    processes.append(process_element)
    return processes


def get_gpu_util(gpu_id):
    """
    Return the overall GPU utilization of the given GPU id
    :param gpu_id: the PCI id of a GPU
    :return: the utilization of the GPU in percent
    """
    smi = get_nvidia_smi_string(gpu_id)
    lines = smi.split("\n")
    utilization = []
    for i in range(len(lines)):
        if lines[i].startswith("    Utilization"):
            end = i
            for j in range(i, len(lines)):
                if lines[j].startswith("    Encoder Stats"):
                    end = j - 1
                    break
            utilization = lines[i:end]
            break
    gpu_util = int([line for line in lines if line.startswith("        Gpu")][0].split(": ")[1].split(" ")[0])
    return gpu_util
