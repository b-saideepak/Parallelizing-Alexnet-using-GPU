#include <stdint.h>

//The datatype of each element - 32 bit signed integer
typedef int32_t DATA;

//Data structure to hold input feature map / weight filters / output feature map
typedef struct _fmap{
  DATA* data;
  int dim1, dim2, dim3, dim4;
}fmap;

//Base class for each layer
class Layer
{
  public:
  double exec_time;
  DATA* weights;
};

class Convolution : public Layer
{
  public:
  int M, C, R, S, Sx, Sy, Px, Py;

  Convolution(int m, int c, int r, int s, int sx, int sy, int px, int py);
  ~Convolution();


  fmap* conv_2d(fmap* input_features);
  fmap* conv2d_IS(fmap* input_features);
  fmap* conv2d_OS(fmap* input_features);
  fmap* conv2d_WS(fmap* input_features);
  fmap* conv2d_optimized(fmap* input_features);
};

class Linear : public Layer
{
  public:
  int M, L;

  Linear(int m, int l);
  ~Linear();
  

  fmap* linear(fmap* input_features);
  fmap* linear_optimized(fmap* input_features);
};

class AlexNet : public Layer
{
  public:
  Convolution** conv_layers;
  Linear** linear_layers;

  AlexNet();
  fmap* forward_pass(fmap* input_features);
};


void relu(fmap* input_features);

void display(fmap* input_features,int,int,int,int,int,int,int,int);

fmap* maxpool_2d(fmap* input_features, int R, int S, int Sx, int Sy);
