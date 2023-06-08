# PCA-Matrix-Addition-With-Unified-Memory
Refer to the program sumMatrixGPUManaged.cu. Would removing the memsets below affect 
performance? If you can, check performance with nvprof or nvvp.
## Aim:
To perform Matrix Addition with Unified Memory and to identify whether removing the memsets
affects the performance or not.

## Procedure:
1. Allocate memory using cudaMallocManaged() function to create a unified memory block.

2. Initialize matrix values in CPU memory.

3. Copy matrix data from CPU memory to unified memory using cudaMemcpy() function with
cudaMemcpyHostToDevice as a direction parameter.

4. Launch a kernel function to perform matrix addition with CUDA syntax.

5. Copy matrix result from unified memory to CPU memory using cudaMemcpy() function
withcudaMemcpyDeviceToHost as a direction parameter.

6. Free memory allocated with cudaFree() function.

## PROGRAM:
## sumMatrixGPUManaged.cu:
```
#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>
void initialData(float *ip, const int size)
{
int i;
for (i = 0; i < size; i++)
{
ip[i] = (float)( rand() & 0xFF ) / 10.0f;
}
return;
}
void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny)
{
float *ia =
A; float *ib
= B;float *ic
= C;
for (int iy = 0; iy < ny; iy++)
{
for (int ix = 0; ix < nx; ix++)
{
ic[ix] = ia[ix] + ib[ix];
}
ia +=
nx; ib
+= nx;ic
+= nx;
}
return;
}
void checkResult(float *hostRef, float *gpuRef, const int N)
{
double epsilon = 1.0E8;bool match = 1;
for (int i = 0; i < N; i++)
{
if (abs(hostRef[i] - gpuRef[i]) > epsilon)
{
match = 0;
printf("host %f gpu %f\n", hostRef[i], gpuRef[i]);
break;
}
}
if (!match)
{ printf("Arrays do not match.\n\n"); }
}
// grid 2D block 2D
 global void sumMatrixGPU(float *MatA, float *MatB, float *MatC, int nx, int ny)
{
unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
unsigned int idx = iy * nx + ix;
if (ix < nx && iy < ny)
{ MatC[idx] = MatA[idx] + MatB[idx]; }
}
int main(int argc, char **argv)
{
printf("%s Starting ", argv[0]);
// set up device
int dev = 0;
cudaDeviceProp deviceProp;
CHECK(cudaGetDeviceProperties(&deviceProp, dev));
printf("using Device %d: %s\n", dev, deviceProp.name);
CHECK(cudaSetDevice(dev));
// set up data size of matrix
int nx, ny;
int ishift = 12;
if (argc > 1) ishift =
atoi(argv[1]);nx = ny = 1 <<
ishift;
int nxy = nx * ny;
int nBytes = nxy * sizeof(float);
printf("Matrix size: nx %d ny %d\n", nx, ny);
// malloc host memory
float *A, *B, *hostRef, *gpuRef;
CHECK(cudaMallocManaged((void **)&A, nBytes));
CHECK(cudaMallocManaged((void **)&B, nBytes));
CHECK(cudaMallocManaged((void **)&gpuRef, nBytes); );
CHECK(cudaMallocManaged((void **)&hostRef, nBytes););
// initialize data at host side
double iStart = seconds();
initialData(A, nxy);
initialData(B, nxy);
double iElaps = seconds() - iStart;
printf("initialization: \t %f sec\n", iElaps);
memset(hostRef, 0, nBytes);
memset(gpuRef, 0, nBytes);
// add matrix at host side for result checks
iStart = seconds();
sumMatrixOnHost(A, B, hostRef, nx,
ny);iElaps = seconds() - iStart;
printf("sumMatrix on host:\t %f sec\n", iElaps);
// invoke kernel at host side
int dimx = 32;
int dimy = 32;
dim3 block(dimx, dimy);
dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
// warm-up kernel, with unified memory all pages will migrate from host to
// device
sumMatrixGPU<<<grid, block>>>(A, B, gpuRef, 1, 1);
// after warm-up, time with unified memory
iStart = seconds();
sumMatrixGPU<<<grid, block>>>(A, B, gpuRef, nx, ny);
CHECK(cudaDeviceSynchronize());
iElaps = seconds() - iStart;
printf("sumMatrix on gpu :\t %f sec <<<(%d,%d), (%d,%d)>>> \n", iElaps, grid.x, grid.y, block.x,
block.y);
// check kernel error
CHECK(cudaGetLastError());
// check device results
checkResult(hostRef, gpuRef, nxy);
// free device global memory
CHECK(cudaFree(A));
CHECK(cudaFree(B));
CHECK(cudaFree(hostRef
));
CHECK(cudaFree(gpuRef)
);
// reset device
CHECK(cudaDeviceReset(
));return (0);
}
```
## OUTPUT:
```
root@SAV-MLSystem:/home/student/Sidd_Lab_Exp_4# nvcc sumMatrixGPUManaged.cu -
osumMatrixGPUManaged
root@SAV-MLSystem:/home/student/Sidd_Lab_Exp_4# nvcc
sumMatrixGPUManaged.curoot@SAV-MLSystem:/home/student/Sidd_Lab_Exp_4#
./sumMatrixGPUManaged
./sumMatrixGPUManaged Starting using Device 0: NVIDIA GeForce GT 710
Matrix size: nx 4096 ny 4096
initialization: 0.471329 sec
sumMatrix on host: 0.034168 sec
sumMatrix on gpu : 0.023536 sec <<<(128,128),
(32,32)>>>root@SAVMLSystem:/home/student/Sidd_Lab_Exp_4#
```
## EXPLANATION:
1. The memsets are used to initialize the hostRef and gpuRef arrays to zero before the computation of the sumMatrixOnHost and sumMatrixGPU functions respectively. Removing these memsets may result in incorrect results if there is any existing data in these arrays.

2. However, since these arrays are allocated using cudaMallocManaged, they are already initialized to zero by default. Therefore, removing the memsets would not affect the correctness of the program.
## Result:
Thus, the Matrix Addition with Unified Memory has been successfully performed and removing the
‘memset’ functions did not have a significant impact on the performance of the program.
