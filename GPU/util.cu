//%%cuda --name util.cu
#include <stdlib.h>
#include <iostream>
#include "util.h"
using namespace std;
__global__ void conv(fmap *input,int *ip,int *weights,int R,int S,fmap *output, int Sx, int Sy,int *op,int Px,int Py){
    unsigned int input_id = (blockIdx.x*gridDim.y + blockIdx.y + blockIdx.z*gridDim.x*gridDim.y)*blockDim.x + threadIdx.x;
    int C,H,W,M,E,F;
    //N = input->dim1;
    C = input->dim2;
    H = input->dim3;
    W = input->dim4;
    M = output->dim2;
    E = output->dim3;
    F = output->dim4;
    H+=2*Py;
    W+=2*Px;
    /*unsigned int weight_id = input_id%(C*R*S);
    int a = weight_id/(R*S);
    weight_id = weight_id%(R*S);
    int b = weight_id/S;
    int c = weight_id%S;*/
    int i = input_id/(M*E*F*C*R*S);
    input_id = input_id%(M*E*F*C*R*S);
    int j = input_id/(E*F*C*R*S);
    input_id = input_id%(E*F*C*R*S);
    int k = input_id/(F*C*R*S);
    input_id = input_id%(F*C*R*S);
    int l = input_id/(C*R*S);
    input_id = input_id%(C*R*S);
    int m = input_id/(R*S);
    input_id = input_id%(R*S);
    int n = input_id/S;
    int o = input_id%S;
    
    int temp = (*(ip + i*C*H*W + m*H*W + (k*Sy + n)*W + (l*Sx + o)))*(*(weights + j*C*R*S + m*R*S + n*S + o));
    atomicAdd((op + i*M*E*F + j*E*F + k*F + l), temp);

   /* printf("Input fmap\n");
    printf("%d %d %d %d\n",N,C,H,W);
    for(int i=0;i<N;i++){
      for(int j=0;j<C;j++){
        for(int k=0;k<H;k++){
          for(int l=0;l<W;l++)
            printf("%3d ",ip[i*C*H*W + j*H*W + k*H + l]);
          printf("\n");
        }
        printf("\n\n");
      }
      printf("\n\n\n");
    }

    printf("Weight fmap\n");
    printf("%d %d %d %d\n",M,C,R,S);
    for(int i=0;i<M;i++){
      for(int j=0;j<C;j++){
        for(int k=0;k<R;k++){
          for(int l=0;l<S;l++)
            printf("%3d ",weights[i*C*R*S + j*R*S + k*S + l]);
          printf("\n");
        }
        printf("\n\n");
      }
      printf("\n\n\n");
    }*/

}

__global__ void RELU(int *ip, int N, int C, int H, int W){
    unsigned int input_id = blockDim.x*blockIdx.x + threadIdx.x;
    int i = input_id/(C*H*W);
    input_id = input_id%(C*H*W);
    int j = input_id/(H*W);
    input_id = input_id%(H*W);
    int k = input_id/(W);
    int l = input_id%W;

    int temp = *(ip + i*C*H*W + j*H*W + k*W + l);
    if(temp<0)
    *(ip + i*C*H*W + j*H*W + k*W + l) = 0;
    
}

__global__ void maxpool(int *ip,int *op,fmap *output,int H,int W,int Sx,int Sy,int R, int S){
    unsigned int input_id = blockDim.x*blockIdx.x + threadIdx.x;
    int C,E,F;
   // N = output->dim1;
    C = output->dim2;
    E = output->dim3;
    F = output->dim4;
    int i = input_id/(C*E*F*R*S);
    input_id = input_id%(C*E*F*R*S);
    int j = input_id/(E*F*R*S);
    input_id = input_id%(E*F*R*S);
    int k = input_id/(F*R*S);
    input_id = input_id%(F*R*S);
    int l = input_id/(R*S);
    input_id = input_id%(R*S);
    int m = input_id/(S);
    int n = input_id%(S);
    //printf("Comparing op[i][j][k][l] with ip\n",*(op + i*C*E*F + j*E*F + k*F + l),*(ip + i*C*H*W + j*H*W + (k*Sy + m) + l*Sx + n));
    atomicMax((op + i*C*E*F + j*E*F + k*F + l),*(ip + i*C*H*W + j*H*W + (k*Sy + m)*W + l*Sx + n));
    
}

__global__ void lineark(int *ip,int *weight,int *op,int N,int M,int L){
    unsigned int input_id = (blockIdx.x*gridDim.y + blockIdx.y + blockIdx.z*gridDim.x*gridDim.y)*blockDim.x + threadIdx.x;
    int i = input_id/(M*L);
    input_id = input_id%(M*L);
    int j = input_id/L;
    int k = input_id%L;

    int temp = (*(ip + i*L + k))*(*(weight + j*L + k));
    atomicAdd((op + i*M + j),temp);
}

__global__ void padding(int *op,int *ip,int N,int C,int H,int W,int Py,int Px){
    unsigned int input_id = (blockIdx.x*gridDim.y + blockIdx.y + blockIdx.z*gridDim.x*gridDim.y)*blockDim.x + threadIdx.x;
    int i = input_id/(C*H*W);
    input_id = input_id%(C*H*W);
    int j = input_id/(H*W);
    input_id = input_id%(H*W);
    int k = input_id/W;
    int l = input_id%W;
    *(op + i*C*(H + 2*Py)*(W + 2*Px) + j*(H + 2*Py)*(W + 2*Px) + (k + Py)*(W + 2*Px) + (l + Px)) = *(ip + i*C*H*W + j*H*W + k*W + l);
}


Convolution::Convolution(int m, int c, int r, int s, int sx, int sy, int px, int py)
{
  M = m;
  C = c;
  R = r;
  S = s;
  Sx = sx;
  Sy = sy;
  Px = px;
  Py = py;
  weights = (DATA*) malloc(M * C * R * S * sizeof(DATA));
  DATA (*temp)[C][R][S] = (DATA (*)[C][R][S])weights;
  for(int i=0; i<M; i++)
    for(int j=0; j<C; j++)
      for(int k=0; k<R; k++)
        for(int l=0; l<S; l++)
          temp[i][j][k][l] = 1;
 //(i*C*R*S+j*R*S+k*S+l)%3;
}

Linear::Linear(int m, int l)
{
  M = m;
  L = l;
  weights = (DATA*) malloc(M * L * sizeof(DATA));
  DATA (*temp)[L] = (DATA (*)[L])weights;
  for(int i=0; i<M; i++)
    for(int j=0; j<L; j++)
      temp[i][j] = (i*L+j)%5;
}

fmap* Convolution::conv_2d(fmap* input_features)
{
  _fmap *input,*output,*output_h;
  output_h = (_fmap*)malloc(sizeof(_fmap));
  int N,C,H,W,E,F,*ip,*weight,*op,*op_h,*d,*i;
  N = input_features->dim1;
  C = input_features->dim2;
  H = input_features->dim3;
  W = input_features->dim4;
  E = (H  - R + 2*Py)/Sy + 1;
  F = (W - S + 2*Px)/Sx + 1;
  output_h->dim1 = N;
  output_h->dim2 = M;
  output_h->dim3 = E;
  output_h->dim4 = F;
  //total_t = N*M*E*F*C*R*S;
  dim3 block(N*M,1,1);
  dim3 grid(C,E*F,R*S);
  output_h->data = (int*)malloc(sizeof(int)*N*M*E*F);
  op_h = output_h->data;
  cudaMalloc(&op,sizeof(int)*N*M*E*F);
  cudaMemset(op,0,sizeof(int)*N*M*E*F);
  cudaMalloc(&weight,sizeof(int)*M*C*R*S);
  cudaMalloc(&output,sizeof(_fmap));
  cudaMalloc(&input,sizeof(_fmap));
  cudaMalloc(&ip,sizeof(int)*N*C*(H + 2*Py)*(W + 2*Px));
  cudaMemcpy(weight,weights,sizeof(int)*M*C*R*S,cudaMemcpyHostToDevice); 
  cudaMemcpy(input,input_features,sizeof(_fmap),cudaMemcpyHostToDevice); 
  cudaMemcpy(output,output_h,sizeof(_fmap),cudaMemcpyHostToDevice);
 /* if(M==96)*/
  
 
   /*Padding the input Matrix*/    
        int *G = (int*)malloc(sizeof(int)*N*C*(H + 2*Py)*(W + 2*Px));/*New Matrix*/
        int (*D)[C][H + 2*Py][W + 2*Px] = (int (*)[C][H + 2*Py][W + 2*Px])G;
 
  memset(D,0,sizeof(int)*N*C*(H + 2*Py)*(W + 2*Px));
  cudaMalloc(&d,sizeof(int)*N*C*(H + 2*Py)*(W + 2*Px));
  cudaMemset(d,0,sizeof(int)*N*C*(H + 2*Py)*(W + 2*Px));
  cudaMalloc(&i,sizeof(int)*N*C*H*W);
  cudaMemcpy(i,input_features->data,sizeof(int)*N*C*H*W,cudaMemcpyHostToDevice);
  dim3 grids(C,H,W);
  dim3 blocks(N,1,1);
  padding<<<grids,blocks>>>(d,i,N,C,H,W,Py,Px);
                                        
  cudaMemcpy(D,d,sizeof(int)*N*C*(H + 2*Py)*(W + 2*Px),cudaMemcpyDeviceToHost); 
  cudaMemcpy(ip,D,sizeof(int)*N*C*(H + 2*Py)*(W + 2*Px),cudaMemcpyHostToDevice); 
  free(input_features->data);
  cudaFree(d);
  cudaFree(i);
       /* Printing the input matrix*/
       /* for(int i=0;i<N;i++){
                for(int j=0;j<C;j++){
                        for(int k=0;k<(H + 2*Py);k++){
                                for(int l=0;l<(W + 2*Px);l++)
                                        printf("%2d ",D[i][j][k][l]);
                                printf("\n");
                                }
                        printf("\n\n");
                        }
                printf("\n\n\n");
                }*/

conv<<<grid,block>>>(input,ip,weight,R,S,output,Sx,Sy,op,Px,Py);
 if ( cudaSuccess != cudaGetLastError() )
    printf( "Error!\n" );
  cudaDeviceSynchronize(); 
  cudaMemcpy(op_h,op,sizeof(int)*N*M*E*F,cudaMemcpyDeviceToHost);
  cudaFree(ip);
  cudaFree(op);
  cudaFree(weight);

    /*printf("Weight fmap\n");
    printf("%d %d %d %d\n",M,C,R,S);
    for(int i=0;i<M;i++){
      for(int j=0;j<C;j++){
        for(int k=0;k<R;k++){
          for(int l=0;l<S;l++)
            printf("%3d ",weights[i*C*R*S + j*R*S + k*S + l]);
          printf("\n");
        }
        printf("\n\n");
      }
      printf("\n\n\n");
    }*/

 /*for(int i=0;i<M;i++)
      for(int j=0;j<C;j++)
        for(int k=0;k<R;k++)
          for(int l=0;l<S;l++)
            printf("%3d ",weights[i*C*R*S + j*R*S + k*S + l]);*/
    /*printf("Output fmap\n");
    printf("%d %d %d %d\n",N,M,E,F);
    for(int i=0;i<N;i++){
      for(int j=0;j<M;j++){
        for(int k=0;k<E;k++){
          for(int l=0;l<F;l++)
            printf("%4d ",op_h[i*M*E*F + j*E*F + k*F + l]);
          printf("\n");
        }
        printf("\n\n");
      }
      printf("\n\n\n");
    }*/
  free(D);
  return output_h;
}

fmap* Linear::linear(fmap* input_features)
{
  _fmap *output_h;
  output_h = (_fmap*)malloc(sizeof(_fmap));
  int N,P,*weight,*ip,*op,*op_h;
  N = input_features->dim1;
  P = input_features->dim2;
  output_h->dim1 = N;
  output_h->dim2 = M;
  output_h->dim3 = 1;
  output_h->dim4 = 1;
  output_h->data = (int*)malloc(sizeof(int)*N*M);
  op_h = output_h->data;
  cudaMalloc(&weight,sizeof(int)*M*L);
  cudaMalloc(&ip,sizeof(int)*N*P);
  cudaMemcpy(weight,weights,sizeof(int)*M*L,cudaMemcpyHostToDevice);
  cudaMemcpy(ip,input_features->data,sizeof(int)*N*P,cudaMemcpyHostToDevice);
  cudaMalloc(&op,sizeof(int)*N*M);
  cudaMemset(op,0,sizeof(int)*N*M);
  
  dim3 block(1,1,1);
  dim3 grid(M,L,N);
  lineark<<<grid,block>>>(ip,weight,op,N,M,L);
  if ( cudaSuccess != cudaGetLastError() )
    printf( "Error!\n" );
  cudaDeviceSynchronize();
  cudaMemcpy(op_h,op,sizeof(int)*N*M,cudaMemcpyDeviceToHost);
  free(input_features->data);
  cudaFree(ip);
  cudaFree(op);
  cudaFree(weight);
  return output_h;
}

void relu(fmap* input_features)
{
	int *ip_h,*ip;
  ip_h = input_features->data;
 
  int N = input_features->dim1;
  int C = input_features->dim2;
  int H = input_features->dim3;
  int W = input_features->dim4;
  cudaMalloc(&ip,sizeof(int)*N*C*H*W);
  cudaMemcpy(ip,ip_h,sizeof(int)*N*C*H*W,cudaMemcpyHostToDevice);
  RELU<<<N*H*C,W>>>(ip,N,C,H,W);
  cudaDeviceSynchronize();
  cudaMemcpy(ip_h,ip,sizeof(int)*N*C*H*W,cudaMemcpyDeviceToHost);
  cudaFree(ip);
  //printf("Relu %d %d %d %d\n",N,C,H,W);
/*if(C==1000){
 for(int i=0;i<N;i++){
    for(int j=0;j<C;j++){
        for(int k=0;k<H;k++){
            for(int l=0;l<W;l++)
              printf("%4d ",*(ip_h + i*C*H*W + j*H*W + k*W + l));
            printf("\n");
        }
        printf("\n\n");
    }
    printf("\n\n\n");
 }
}*/
/* if(C==1000){
 for(int i=0;i<N;i++)
    for(int j=0;j<C;j++)
        for(int k=0;k<H;k++)
            for(int l=0;l<W;l++)
              printf("%4d ",*(ip_h + i*C*H*W + j*H*W + k*W + l));
 
     printf("\n\n\n\n"); 
 }       */
         
  
}

fmap* maxpool_2d(fmap* input_features, int R, int S, int Sx, int Sy)
{
	_fmap *output,*output_h; int N,C,H,W,E,F,*op,*op_h,*ip;
  output_h = (_fmap*)malloc(sizeof(_fmap));
  N = input_features->dim1;
  C = input_features->dim2;
  H = input_features->dim3;
  W = input_features->dim4; 
  E = (H  - R)/Sy + 1;
  F = (W - S )/Sx + 1;
  output_h->dim1 = N;
  output_h->dim2 = C;
  output_h->dim3 = E;
  output_h->dim4 = F;
  output_h->data = (int*)malloc(sizeof(int)*N*C*E*F);
  op_h = output_h->data;
  cudaMalloc(&output,sizeof(_fmap));
  cudaMalloc(&op,sizeof(int)*N*C*E*F);
   cudaMemset(op, 0, sizeof(int)*N*C*E*F);
  cudaMalloc(&ip,sizeof(int)*N*C*H*W);
  cudaMemcpy(output,output_h,sizeof(_fmap),cudaMemcpyHostToDevice);
  cudaMemcpy(ip,input_features->data,sizeof(int)*N*C*H*W,cudaMemcpyHostToDevice);
  maxpool<<<N*E*F*R*S,C>>>(ip,op,output,H,W,Sx,Sy,R,S);
  if ( cudaSuccess != cudaGetLastError() )
    printf( "Error!\n" );
  cudaDeviceSynchronize();
  cudaMemcpy(op_h,op,sizeof(int)*N*C*E*F,cudaMemcpyDeviceToHost);
  cudaFree(ip);
  cudaFree(op);
 
  //printf("Maxpool %d %d %d %d\n",N,C,E,F);

 /*for(int i=0;i<N;i++){
    for(int j=0;j<C;j++){
        for(int k=0;k<E;k++){
            for(int l=0;l<F;l++)
              printf("%4d ",*(op_h + i*C*E*F + j*E*F + k*F + l));
            printf("\n");
        }
        printf("\n\n");
    }
    printf("\n\n\n");
 }*/
     
  free(input_features->data);
  return output_h;
}

void display(fmap* input_features)
{
        int N,C,H,W,*dat;
        N = input_features->dim1;
        C = input_features->dim2;
        H = input_features->dim3;
        W = input_features->dim4;
        printf("Size %d\n",N*C*H*W);
        dat = input_features->data;
        for(int i=0;i<N;i++)
                for(int j=0;j<C;j++)
                        for(int k=0;k<H;k++)
                                for(int l=0;l<W;l++)
                                        printf("%d\n",*(dat + i*C*H*W + j*H*W + k*W + l));
        printf("\n\n\n");
}


AlexNet::AlexNet()
{
  conv_layers = (Convolution**) malloc(5 * sizeof(Convolution*));

  Convolution *conv;
  conv = new Convolution(96, 3, 11, 11, 4, 4, 2, 2);
  conv_layers[0] = conv;
  conv = new Convolution(256, 96, 5, 5, 1, 1, 2, 2);
  conv_layers[1] = conv;
  conv = new Convolution(384, 256, 3, 3, 1, 1, 1, 1);
  conv_layers[2] = conv;
  conv = new Convolution(384, 384, 3, 3, 1, 1, 1, 1);
  conv_layers[3] = conv;
  conv = new Convolution(256, 384, 3, 3, 1, 1, 1, 1);
  conv_layers[4] = conv;

  linear_layers = (Linear**) malloc(3 * sizeof(Linear*));

  Linear *linear;
  linear = new Linear(4096, 9216);
  linear_layers[0] = linear;
  linear = new Linear(4096, 4096);
  linear_layers[1] = linear;
  linear = new Linear(1000, 4096);
  linear_layers[2] = linear;
}

fmap* AlexNet::forward_pass(fmap* input_features)
{
  clock_t start, end,t;
  fmap* temp = input_features;
 
  start = clock();
  temp = conv_layers[0]->conv_2d(temp);
  relu(temp);
  temp = maxpool_2d(temp, 3, 3, 2,2);
  t = clock();
  cout<<"C1 exec "<<double(t - start) / double(CLOCKS_PER_SEC)<<endl;
  
  temp = conv_layers[1]->conv_2d(temp);
  relu(temp);
  temp = maxpool_2d(temp, 3, 3, 2, 2);
  end = clock();
  cout<<"C2 exec "<<double(end - t) / double(CLOCKS_PER_SEC)<<endl;
  
  temp = conv_layers[2]->conv_2d(temp);
  relu(temp);
  t = clock();
  cout<<"C3 exec "<<double(t - end) / double(CLOCKS_PER_SEC)<<endl;
  
  temp = conv_layers[3]->conv_2d(temp);
  relu(temp);
  end = clock();
  cout<<"C4 exec "<<double(end - t) / double(CLOCKS_PER_SEC)<<endl;
  
  temp = conv_layers[4]->conv_2d(temp);
  relu(temp);
  temp = maxpool_2d(temp, 3, 3, 2, 2);
  t = clock();
  cout<<"C5 exec "<<double(t - end) / double(CLOCKS_PER_SEC)<<endl;
  
  int lin_dim = temp->dim2 * temp->dim3 * temp->dim4;
  temp->dim2 = lin_dim;
  temp->dim3 = temp->dim4 = 1;
  t = clock();
  temp = linear_layers[0]->linear(temp);
  relu(temp);
  end = clock();
  cout<<"L1 exec "<<double(end - t) / double(CLOCKS_PER_SEC)<<endl;
  
  temp = linear_layers[1]->linear(temp);
  relu(temp);
  t = clock();
  cout<<"L2 exec "<<double(t - end) / double(CLOCKS_PER_SEC)<<endl;
  
  temp = linear_layers[2]->linear(temp);
  relu(temp);
  end = clock();
  cout<<"L3 exec "<<double(end - t) / double(CLOCKS_PER_SEC)<<endl;
	  
  end = clock();
  exec_time = double(end-start) / double(CLOCKS_PER_SEC);
  display(temp);
  return temp;
}
