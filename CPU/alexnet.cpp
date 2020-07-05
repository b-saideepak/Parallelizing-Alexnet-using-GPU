#include <iostream>
#include <stdlib.h>
#include "util.h"

#define N 1
#define C 3
#define H 227
#define W 227

int main()
{
  AlexNet net;
  
  fmap input;
  input.dim1 = N;
  input.dim2 = C;
  input.dim3 = H;
  input.dim4 = W;
  input.data = (DATA*) malloc(N * C * H * W * sizeof(DATA));

  DATA (*temp)[C][H][W] = (DATA (*)[C][H][W])input.data;

  for(int i=0; i<N; i++)
    for(int j=0; j<C; j++)
      for(int k=0; k<H; k++)
        for(int l=0; l<W; l++)
          temp[i][j][k][l] = (i*C*H*W+j*H*W+k*W+l)%2;

  fmap* output = net.forward_pass(&input);

  std::cout << "Total exec "<<net.exec_time << std::endl;

  return 0;
}
