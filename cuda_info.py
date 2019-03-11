from numba import cuda

def main():
     
    if cuda.is_available():
        dev_no = cuda.cudadrv.driver.Device(0).id
        print(cuda.list_devices())
        print(cuda.cudadrv.driver.Device(dev_no).compute_capability)
        print(cuda.cudadrv.driver.Device(dev_no).name)
    else:
        print("no GPU detected")


main()


              
