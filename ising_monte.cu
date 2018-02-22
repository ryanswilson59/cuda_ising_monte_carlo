#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <stdlib.h>

#define width 16
#define height 16
#define N 4
#define runs (10000)

__device__ int dx=width/N;
__device__ int dy=height/N;

int dx_h=width/N;
int dy_h=width/N;

__device__ const int mxar[4]={-1,1,0,0};
__device__ const int myar[4]={0,0,-1,1};

float beta_t=0.3;


//https://stackoverflow.com/questions/18501081/generating-random-number-within-cuda-kernel-in-a-varying-range
//source of curand
__global__ void init_value(int *grd,int val,int pitch){
	int gx=16*blockIdx.x+threadIdx.x;int gy=16*blockIdx.y+threadIdx.y;
	int loc=gx*pitch+gy;
	grd[loc]=val;
}


__global__ void setup_kernel ( curandState * state, unsigned long seed )
{
    int id = threadIdx.x+N*threadIdx.y;
    curand_init ( seed+id, id, 0, &state[id] );
}

__global__ void monte_run(int *grd,int pitch,int *acc,int offx,int offy,float beta,curandState* globalState){
	int x=threadIdx.x; int y=threadIdx.y;
	int gx=x*dx+offx;int gy=y*dy+offy;

	int id = threadIdx.x+N*threadIdx.y;

	curandState localState=globalState[id];

	int lsum=0;
	for(int i=0;i<4;i++){
		int tx=gx+mxar[i];
		int ty=gy+myar[i];
		if(tx>=0 and tx<width and ty>=0 and ty<height){
			lsum+=grd[ty*pitch+tx];
		}
	}
	float p;
	if(lsum*grd[gy*pitch+gx]>0){
		p=-lsum*grd[gy*pitch+gx]*beta;
		p=expf(p);
	}else{
		p=1.0;
	}
	float rnd=curand_uniform(&localState);
	globalState[id]=localState;
	if (rnd<p){
		acc[id]=lsum*grd[gy*pitch+gx];
		grd[gy*pitch+gx]=-grd[gy*pitch+gx];
	}else{
		acc[id]=0;
	}
}

// this only works for one block, see below for more blocks implementation
//https://gist.github.com/wh5a/4424992

//N needs to be even otherwise "fun times"
__global__ void reduce_dif(int *nums, int *acc,int ind){
	__shared__ int partial_sum[N*N];
	int x=threadIdx.x;
	if (x<N*N){
		partial_sum[x]=nums[x];
	}else{
		partial_sum[x]=0;
	}
	for(int stride=N*N>>1;stride>=1;stride>>=1){
		if (stride!=1 and stride%2==1){
			stride+=1;
		}
		__syncthreads();
		if(x<stride){
			partial_sum[x]+=partial_sum[x+stride];
		}
	}
	if (x==0){
		acc[ind]=partial_sum[0];
	}
}

int main(){

	//float* d_A;
	//cudaMalloc(&d_A, size);

	//int8_t init_grid[width][height];

	int* devPtr;
	size_t pitch;

	cudaMallocPitch(&devPtr,&pitch,width*sizeof(int),height);
	// want to use cudaMemset2D but no source found
	//No source available for "memset32() at 0xb5d7e8"
	cudaMemset(devPtr,0,pitch*height*sizeof(int));

	dim3 set_grid_size((width-1)/16+1,(height-1)/16+1);
	dim3 set_block_size(16,16);
	init_value<<<set_grid_size,set_block_size>>>(devPtr,1,pitch);
	//pattern to access is rownum*pitch+colnum
	//cudaMemcpy2D(devPtr,pitch,init_grid,width*sizeof(int8_t),width*sizeof(int8_t),height,cudaMemcpyHostToDevice);

	const int accsize=sizeof(int)*runs;
	int* accPtr;
	cudaMalloc(&accPtr,accsize);
	cudaMemset(accPtr,0,runs*sizeof(int));

	int *accHost;
	accHost= (int *) malloc(runs*sizeof(int));

	int* monte_acc_a;
	int* monte_acc_b;
	cudaMalloc(&monte_acc_a,N*N*sizeof(int));
	cudaMalloc(&monte_acc_b,N*N*sizeof(int));

	dim3 gridsize(1,1);
	dim3 blocksize(N,N);

	curandState* devStates;
	cudaMalloc( &devStates, N*N*sizeof( curandState ) );

	setup_kernel <<< gridsize, blocksize >>> ( devStates, time(NULL) );

	int mdx=dx_h+width%N;
	int mdy=dy_h+height%N;

	srand(time(NULL));

	int tdx= rand()%mdx;
	int tdy= rand()%mdy;

	cudaDeviceSynchronize();
	monte_run <<< gridsize,blocksize >>>(devPtr,pitch,monte_acc_a,tdx,tdy,beta_t,devStates);
	tdx= rand()%mdx;
	tdy= rand()%mdy;
	cudaDeviceSynchronize();
	for(int i=0;i<(runs/2)-1;i++){
		reduce_dif <<< 1,N*N >>>(monte_acc_a,accPtr,2*i);
		monte_run <<< gridsize,blocksize >>>(devPtr,pitch,monte_acc_b,tdx,tdy,beta_t,devStates);
		tdx= rand()%mdx;
		tdy= rand()%mdy;
		cudaDeviceSynchronize();
		reduce_dif <<< 1,N*N >>>(monte_acc_b,accPtr,2*i+1);
		monte_run <<< gridsize,blocksize >>>(devPtr,pitch,monte_acc_a,tdx,tdy,beta_t,devStates);
		tdx= rand()%mdx;
		tdy= rand()%mdy;
		cudaDeviceSynchronize();
	}
	reduce_dif <<< 1,N*N >>>(monte_acc_a,accPtr,runs-2);
	monte_run <<< gridsize,blocksize >>>(devPtr,pitch,monte_acc_b,tdx,tdy,beta_t,devStates);
	tdx= rand()%mdx;
	tdy= rand()%mdy;
	cudaDeviceSynchronize();
	reduce_dif <<< 1,N*N >>>(monte_acc_b,accPtr,runs-1);
	cudaMemcpy(accHost,accPtr,accsize,cudaMemcpyDeviceToHost);

	cudaFree(devStates);
	cudaFree(monte_acc_a);
	cudaFree(monte_acc_b);
	cudaFree(accPtr);
	cudaFree(devPtr);

	FILE *fp;
	fp=fopen("text.txt","w");
	if(fp == NULL)
	    exit(-1);


	for (int i=0;i<runs;i++){
		fprintf(fp,"%d\n",accHost[i]);
	}
	fclose(fp);

	free(accHost);
	return EXIT_SUCCESS;
}

