#include "stdio.h"
#include "cuda_runtime.h"

// output given cudaDeviceProp
void OutputSpec( const cudaDeviceProp sDevProp )
{
  printf( "Device name: %s\n", sDevProp.name );
  printf( "Device memory: %d\n", sDevProp.totalGlobalMem );
  printf( " Memory per-block: %d\n", sDevProp.sharedMemPerBlock );
  printf( " Register per-block: %d\n", sDevProp.regsPerBlock );
  printf( " Warp size: %d\n", sDevProp.warpSize );
  printf( " Memory pitch: %d\n", sDevProp.memPitch );
  printf( " Constant Memory: %d\n", sDevProp.totalConstMem );
  printf( "Max thread per-block: %d\n", sDevProp.maxThreadsPerBlock );
  printf( "Max thread dim: ( %d, %d, %d )\n", sDevProp.maxThreadsDim[0], sDevProp.maxThreadsDim[1], sDevProp.maxThreadsDim[2] );
  printf( "Max grid size: ( %d, %d, %d )\n", sDevProp.maxGridSize[0], sDevProp.maxGridSize[1], sDevProp.maxGridSize[2] );
  printf( "Ver: %d.%d\n", sDevProp.major, sDevProp.minor );
  printf( "Clock: %d\n", sDevProp.clockRate );
  printf( "textureAlignment: %d\n", sDevProp.textureAlignment );
}

int main()
{
  // part1, check the number of device
  int  iDeviceCount = 0;
  cudaGetDeviceCount( &iDeviceCount );
  printf( "Number of GPU: %d\n\n", iDeviceCount );

  if( iDeviceCount == 0 )
  {
    printf( "No supported GPU\n" );
    return 0;
  }

  // part2, output information of each device
  for( int i = 0; i < iDeviceCount; ++ i )
  {
    printf( "\n=== Device %i ===\n", i );
    cudaDeviceProp  sDeviceProp;
    cudaGetDeviceProperties( &sDeviceProp, i );
    OutputSpec( sDeviceProp );
  }
}