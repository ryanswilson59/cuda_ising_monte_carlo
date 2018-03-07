#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <stdlib.h>


#define MIN(a,b) ({typeof(a)_a=(a);typeof(a)_b=(b);_a<_b?_a:_b;})
#define MAX(a,b) ({typeof(a)_a=(a);typeof(a)_b=(b);_a>_b?_a:_b;})

__device__ const int mxar[4]={-1,1,0,0};
__device__ const int myar[4]={0,0,-1,1};

//TODO: fix mysterious size in range [17-32] bug with really large values
// hiding in devPtr after init_value, until then don't use h/w in that range
// has nothing to do with that range, kinda random though, also never observed on 8? longer sims seem more likely to be affected?

//float beta_t=0.3;


//https://stackoverflow.com/questions/18501081/generating-random-number-within-cuda-kernel-in-a-varying-range
//source of curand
__global__ void init_value(int *grd,int val,int pitch,int w,int h){
	int gx=16*blockIdx.x+threadIdx.x;int gy=16*blockIdx.y+threadIdx.y;
	if (gx>=0 and gx<w and gy>=0 and gy<h){
		int loc=gx*pitch+gy;
		grd[loc]=val;
	}
	//__syncthreads();
	//if(gx>=0 and gx<w and gy>=0 and gy<h){
	//	if(MAX(grd[gy*pitch+gx],-grd[gy*pitch+gx])>2){
	//		grd[gy*pitch+gx]+=1;
	//	}
	//}
	__syncthreads();
}


__global__ void setup_kernel ( curandState * state, unsigned long seed,int n )
{
    int id = threadIdx.x+n*threadIdx.y;
    curand_init ( seed+id, id, 0, &state[id] );
    __syncthreads();
}

__global__ void monte_run(int *grd,int pitch,int *acc,int offx,int offy,float beta,curandState* globalState,int dx, int dy,int n,int w,int h){
	int x=threadIdx.x; int y=threadIdx.y;
	int gx=x*dx+offx;int gy=y*dy+offy;

	int id = threadIdx.x+n*threadIdx.y;

	curandState localState=globalState[id];
	int lsum=0;
	for(int i=0;i<4;i++){
		int tx=gx+mxar[i];
		int ty=gy+myar[i];
		if(tx>=0 and tx<w and ty>=0 and ty<h){
			lsum+=grd[ty*pitch+tx];
			//if(MAX(grd[ty*pitch+tx],-grd[ty*pitch+tx])>2){
			//	grd[ty*pitch+tx]+=1;
			//}
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
	__syncthreads();
}

// this only works for one block, see below for more blocks implementation
//https://gist.github.com/wh5a/4424992

__global__ void reduce_dif(int *nums, int *acc,int ind, int n){
	int x=threadIdx.x;
	for(int stride=(n*n+1)>>1;stride>=1;stride=(stride+1)>>1){
		__syncthreads();
		if(x<2*(stride>>1)){
			nums[x]+=nums[x+stride];
		}
		if(stride==1){
			stride=0;
		}
	}
	if (x==0){
		acc[ind]=nums[0];
	}
	__syncthreads();
}

int* run_sim(float beta,int w,int h,int steps){
	int *hist;
	hist= (int *) malloc(steps*sizeof(int));

	int* devPtr;
	size_t pitch;
	//cudaMallocPitch(&devPtr,&pitch,w*sizeof(int),h);
	cudaMalloc(&devPtr,w*h*sizeof(int));
	cudaMemset(devPtr,0,w*h*sizeof(int));
	dim3 set_grid_size((w-1)/16+1,(h-1)/16+1);
	dim3 set_block_size(16,16);
	init_value<<<set_grid_size,set_block_size>>>(devPtr,1,w,w,h);
	cudaDeviceSynchronize();

	const int accsize=sizeof(int)*steps;
	int* accPtr;
	cudaMalloc(&accPtr,accsize);
	cudaMemset(accPtr,0,steps*sizeof(int));

	int n=MIN(16,MIN(w/2,h/2));

	int* monte_acc_a;
	int* monte_acc_b;
	cudaMalloc(&monte_acc_a,n*n*sizeof(int));
	cudaMalloc(&monte_acc_b,n*n*sizeof(int));
	dim3 gridsize(1,1);
	dim3 blocksize(n,n);

	curandState* devStates;
	cudaMalloc( &devStates, n*n*sizeof( curandState ) );
	setup_kernel <<< gridsize, blocksize >>> ( devStates, time(NULL) ,n);

	int dx=w/n;
	int dy=h/n;

	int mdx=dx+w%n;
	int mdy=dy+h%n;
	srand(time(NULL));
	int tdx= rand()%mdx;
	int tdy= rand()%mdy;

	cudaDeviceSynchronize();

	monte_run <<< gridsize,blocksize >>>(devPtr,w,monte_acc_a,tdx,tdy,beta,devStates,dx,dy,n,w,h);
	tdx= rand()%mdx;
	tdy= rand()%mdy;
	cudaDeviceSynchronize();
	for(int i=0;i<(steps/2)-1;i++){

		reduce_dif <<< 1,n*n >>>(monte_acc_a,accPtr,2*i,n);
		monte_run <<< gridsize,blocksize >>>(devPtr,w,monte_acc_b,tdx,tdy,beta,devStates,dx,dy,n,w,h);
		tdx= rand()%mdx;
		tdy= rand()%mdy;
		cudaDeviceSynchronize();

		reduce_dif <<< 1,n*n >>>(monte_acc_b,accPtr,2*i+1,n);
		monte_run <<< gridsize,blocksize >>>(devPtr,w,monte_acc_a,tdx,tdy,beta,devStates,dx,dy,n,w,h);
		tdx= rand()%mdx;
		tdy= rand()%mdy;
		cudaDeviceSynchronize();
	}
	reduce_dif <<< 1,n*n >>>(monte_acc_a,accPtr,steps-2,n);
	monte_run <<< gridsize,blocksize >>>(devPtr,w,monte_acc_b,tdx,tdy,beta,devStates,dx,dy,n,w,h);
	tdx= rand()%mdx;
	tdy= rand()%mdy;
	cudaDeviceSynchronize();

	reduce_dif <<< 1,n*n >>>(monte_acc_b,accPtr,steps-1,n);
	cudaMemcpy(hist,accPtr,accsize,cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	cudaFree(devStates);
	cudaFree(monte_acc_a);
	cudaFree(monte_acc_b);
	cudaFree(accPtr);
	cudaFree(devPtr);

	return hist;
}

int main(int argc, char **argv){

	float t_min;
	float t_max;
	float t_steps;

	int steps;
	int * size_l;
	int l;
	size_l=(int*)malloc((MAX(1,argc-5))*sizeof(int));

	if (argc>=6){
		l=argc-5;
		for(int i=5;i<argc;i++){
			size_l[i-5]=atoi(argv[i]);
		}
		steps=atoi(argv[1]);
		t_min=atof(argv[2]);
		t_max=atof(argv[3]);
		t_steps=atoi(argv[4]);

	}else{
		l=1;
		steps=(100);
		t_min=3;
		t_max=5;
		t_steps=5;
		size_l[0]=4;
	}
	printf("test");
	for(int i=0;i<l;i++){
		int h=size_l[i];int w=size_l[i];
		for(int t_step=0;t_step<t_steps;t_step++){
			float t=t_min+t_step*(t_max-t_min)/(t_steps-1);
			float beta=1/t;
			int *res=run_sim(beta,w,h,steps);

			FILE *fp;
			char *fname=(char*)malloc(100*sizeof(char));
			sprintf(fname,"data/ising_l_%d_t_%f.txt",w,t);
			fp=fopen(fname,"w");
			if(fp == NULL){
			    exit(-1);
			}
			for (int i=0;i<steps;i++){
				fprintf(fp,"%d\n",res[i]);
			}
			fclose(fp);

			free(res);
			printf("\r finished: t step=%d l=%d           ",t_step,w);
		}
	}
	free(size_l);
	return EXIT_SUCCESS;
}


