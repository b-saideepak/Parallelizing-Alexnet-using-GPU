#include <stdlib.h>
#include <iostream>
#include "util.h"
#include<string.h>
using namespace std;
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
  //(i*C*R*S+j*R*S+k*S+l)%2;
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
  	int N,C,H,W,E,F; int *temp1,*G;

	/*Extract the features of input_map*/
	N = input_features->dim1;
	C = input_features->dim2;
	H = input_features->dim3;
	W = input_features->dim4;
	temp1 = input_features->data;
	 //printf("Convolution %d %d %d %d\n",N,C,H,W);

	/*Transform input_data into multi-dimensional*/
	DATA (*ip_data)[C][H][W] = (DATA (*)[C][H][W])temp1;

	/*Output Matrix*/
	E = (H  - R + 2*Py)/Sy + 1;
	F = (W - S + 2*Px)/Sx + 1;
	_fmap *output_map = (struct _fmap*)malloc(sizeof(struct _fmap));
	output_map->dim1 = N;
	output_map->dim2 = M;
	output_map->dim3 = E;
	output_map->dim4 = F;
  	output_map->data = ( DATA *)malloc(sizeof(int) * N * M * E * F);
	DATA (*out)[M][E][F] = (DATA (*)[M][E][F])output_map->data;
	memset(out,0,sizeof(int)*N*M*E*F);
	/*Transforming the weight-matrix*/
	DATA (*temp)[C][R][S] = (DATA (*)[C][R][S])weights;
	
	/*printing the weight matrix	
	for(int i=0;i<M;i++){
		for(int j=0;j<C;j++){
			for(int k=0;k<R;k++){
				for(int l=0;l<S;l++)
					printf("%4d ",temp[i][j][k][l]);
				printf("\n");
				}
			printf("\n\n");
			}
		printf("\n\n\n");
		}*/
	/*Padding the input Matrix*/	
	G = (int*)malloc(sizeof(int)*N*C*(H + 2*Py)*(W + 2*Px));/*New Matrix*/
  	int (*D)[C][H + 2*Py][W + 2*Px] = (int (*)[C][H + 2*Py][W + 2*Px])G;
	memset(D,0,sizeof(int)*N*C*(H + 2*Py)*(W + 2*Px));
  	for(int i=0;i<N;i++)
		for(int j=0;j<C;j++)
			for(int k=Py;k<(H + Py);k++)
				for(int l=Px;l<(W + Py);l++)
					D[i][j][k][l] = ip_data[i][j][k-Py][l-Px];
					free(input_features->data);

	/*Printing the input matrix
	for(int i=0;i<N;i++){
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
	/*Convolution*/
	for(int i=0;i<N;i++)
		for(int j=0;j<M;j++)
			for(int k=0;k<E;k++)
				for(int l=0;l<F;l++)
					for(int m=0;m<C;m++)
						for(int n=0;n<R;n++)
							for(int o=0;o<S;o++)
								out[i][j][k][l]+=(D[i][m][k*Sy + n][l*Sx + o]*temp[j][m][n][o]);
	/*Printing the Output Matrix
	for(int i=0;i<N;i++){
		for(int j=0;j<M;j++){
			for(int k=0;k<E;k++){
				for(int l=0;l<F;l++)
					printf("%4d ",out[i][j][k][l]);
				printf("\n");
				}
			printf("\n\n");
			}
		printf("\n\n\n");
		}*/
	 //printf("%d %d %d %d\n",N,M,E,F);


	free(D);
  return output_map;
}

fmap* Linear::linear(fmap* input_features)
{
	int N,P,sum=0; 
	/*Input Matrix*/
	N = input_features->dim1;
	P = input_features->dim2;
	DATA (*ip)[P] = (DATA (*)[P])input_features->data;

	/*Weight Matrix*/
	DATA (*temp)[L] = (DATA (*)[L])weights;

	/*Output Matrix*/
	_fmap *output_map = (struct _fmap*)malloc(sizeof(struct _fmap));
	output_map->dim1 = N;
	output_map->dim2 = M;
	output_map->dim3 = 1;
	output_map->dim4 = 1;
  	output_map->data = ( DATA *)malloc(sizeof(int) * N * M);
	DATA (*out)[M] = (DATA (*)[M])output_map->data;
	memset(out,0,sizeof(int)*N*M);
	for(int i=0;i<N;i++)
		for(int j=0;j<M;j++){
			sum=0;
			for(int k=0;k<L;k++){
				sum+=(ip[i][k]*temp[j][k]);
			//	printf("Multiplying %d and %d %d\n",ip[i][k],temp[j][k],sum);
			}
			out[i][j] = sum;
		}
	free(input_features->data);
	//printf("Linear %d %d\n",N,M);
	
  return output_map;
}

void relu(fmap* input_features)
{
	int N,C,H,W; int *temp;

	N = input_features->dim1;
	C = input_features->dim2;
	H = input_features->dim3;
	W = input_features->dim4;
	temp = input_features->data;
	DATA (*data)[C][H][W] = (DATA (*)[C][H][W])temp;

	for(int i=0;i<N;i++)
		for(int j=0;j<C;j++)
			for(int k=0;k<H;k++)
				for(int l=0;l<W;l++)
					if(data[i][j][k][l]<0)
						data[i][j][k][l] = 0;
	 //printf("Relu %d %d %d %d\n",N,C,H,W);

}

fmap* maxpool_2d(fmap* input_features, int R, int S, int Sx, int Sy)
{
	int N,C,H,W,E,F,max=0,t=0; int *temp;

	N = input_features->dim1;
	C = input_features->dim2;
	H = input_features->dim3;
	W = input_features->dim4;
	temp = input_features->data;
	DATA (*ip)[C][H][W] = (DATA (*)[C][H][W])temp;
       	/*Output Matrix*/
	E = (H - R )/Sy + 1;
	F = (W - S )/Sx + 1;
	_fmap *output_map = (struct _fmap*)malloc(sizeof(struct _fmap));
	output_map->dim1 = N;
	output_map->dim2 = C;
	output_map->dim3 = E;
	output_map->dim4 = F;
  	output_map->data = ( DATA *)malloc(sizeof(int) * N * C * E * F);
	DATA (*out)[C][E][F] = (DATA (*)[C][E][F])output_map->data;
	//printf("Maxpool %d %d %d %d\n",N,C,E,F);
  

	/*Maxpool 2d*/
	for(int i=0;i<N;i++)
		for(int k=0;k<E;k++)
			for(int l=0;l<F;l++)
				for(int m=0;m<C;m++){
					max = 0;
					for(int n=0;n<R;n++)
						for(int o=0;o<S;o++){
							t = ip[i][m][k*Sy + n][l*Sx + o];
							if(t > max)
								max = t;
							}
				out[i][m][k][l] = max;
	
				}
       /*Printing the Output Matrix
	for(int i=0;i<N;i++){
		for(int j=0;j<C;j++){
			for(int k=0;k<E;k++){
				for(int l=0;l<F;l++)
					printf("%4d ",out[i][j][k][l]);
				printf("\n");
				}
			printf("\n\n");
			}
		printf("\n\n\n");
		}*/
	
       //printf("%d %d %d %d\n",N,C,E,F);
       free(ip);
       return output_map;
}

void display(fmap* input_features)
{
	int N,C,H,W;
	N = input_features->dim1;
	C = input_features->dim2;
	H = input_features->dim3;
	W = input_features->dim4;
	
	DATA (*out)[C][H][W] = (DATA (*)[C][H][W])input_features->data;
	for(int i=0;i<N;i++)
                for(int j=0;j<C;j++)
                        for(int k=0;k<H;k++)
                                for(int l=0;l<W;l++)
					printf("%d\n",out[i][j][k][l]);
	printf("\n");
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
