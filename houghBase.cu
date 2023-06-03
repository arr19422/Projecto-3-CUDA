/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   :
 To build use  : make
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <corecrt_math_defines.h>
#include <cuda.h>
#include <string.h>
#include <tuple>
#include <vector>
#include "pgm.h"
#include "CImg.h"

using namespace std;

const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;
const int rBins = 100;
const float radInc = degreeInc * M_PI / 180;
//*****************************************************************
// The CPU function returns a pointer to the accummulator
void CPU_HoughTran (unsigned char *pic, int w, int h, vector<tuple<int, int, int, int, int>> *acc)
{
  float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;  //(w^2 + h^2)/2, radio max equivalente a centro -> esquina
  //*acc = new int[rBins * degreeBins];            //el acumulador, conteo depixeles encontrados, 90*180/degInc = 9000
  //memset (*acc, 0, sizeof (int) * rBins * degreeBins); //init en ceros
  // inicializar el acumulador en ceros
  //*acc = vector<tuple<int, int, int>>(rBins * degreeBins, make_tuple(0, 0, 0));
  int xCent = w / 2;
  int yCent = h / 2;
  float rScale = 2 * rMax / rBins;

  //cout << "rMax: " << rMax << endl;
  //cout << "xCent: " << xCent << endl;
  //cout << "yCent: " << yCent << endl;
  //cout << "rScale: " << rScale << endl;

  bool continueloop = true;

  for (int i = 0; i < w; i++) { //por cada pixel
    for (int j = 0; j < h; j++) //...
    {
      int idx = j * w + i;
      if (pic[idx] > 10 && continueloop) //si pasa thresh, entonces lo marca
      {
        int xCoord = i - xCent;
        int yCoord = yCent - j;  // y-coord has to be reversed
        //float theta = 0;         // actual angle
        //cout << "xCoord: " << xCoord << endl;
        //cout << "yCoord: " << yCoord << endl;
        // recorre de 0 a pi, pero por indices
        for (int tIdx = 0; tIdx < degreeBins; tIdx++) //add 1 to all lines in that pixel
        {
          //cout << "------------------------------------" << endl;
          float theta = tIdx * radInc;
            
          //cout << "theta: " << theta << endl;
          float r = xCoord * cos (theta) + yCoord * sin (theta);
          //cout << "r: " << r << endl;
          //xint rIdx = (r + rMax) / rScale;
          int rIdx = static_cast<int>((r + rMax) / rScale);
          //float rIdx = (r + rMax) / rScale;
          //cout << "rIdx: " << rIdx << endl;
          //cout << "acc index: " << rIdx * degreeBins + tIdx << endl;
          // +1 para este radio r y este theta
          acc->at(rIdx * degreeBins + tIdx) = make_tuple(rIdx, tIdx, xCoord, yCoord, get<4>(acc->at(rIdx * degreeBins + tIdx)) + 1);
          //cout << "acc: " << get<4>(acc->at(rIdx * degreeBins + tIdx)) << endl;
          //xtheta += radInc;
          //x//cout << "theta: " << theta << endl;
        }
        /*if (i == (w/2)) {
          continueloop = false;
        }*/
        //continueloop = false;
      }
    }
  }
}
bool compareTuples(const tuple<int, int, int, int, int>& a, const tuple<int, int, int, int, int>& b) {
    return get<4>(a) > get<4>(b); // Sort in descending order
}

//*****************************************************************
// TODO usar memoria constante para la tabla de senos y cosenos
// inicializarlo en main y pasarlo al device
//__constant__ float d_Cos[degreeBins];
//__constant__ float d_Sin[degreeBins];

//*****************************************************************
//TODO Kernel memoria compartida
// __global__ void GPU_HoughTranShared(...)
// {
//   //TODO
// }
//TODO Kernel memoria Constante
// __global__ void GPU_HoughTranConst(...)
// {
//   //TODO
// }

// GPU kernel. One thread per image pixel is spawned.
// The accummulator memory needs to be allocated by the host in global memory
__global__ void GPU_HoughTran (unsigned char *pic, int w, int h, int *acc, float rMax, float rScale, float *d_Cos, float *d_Sin)
{
  int gloID = blockIdx.x * blockDim.x + threadIdx.x;
  if (gloID > w * h) return;      // in case of extra threads in block

  int xCent = w / 2;
  int yCent = h / 2;

  //TODO explicar bien bien esta parte. Dibujar un rectangulo a modo de imagen sirve para visualizarlo mejor
  int xCoord = gloID % w - xCent;
  int yCoord = yCent - gloID / w;

  //TODO eventualmente usar memoria compartida para el acumulador

  if (pic[gloID] > 0)
    {
      for (int tIdx = 0; tIdx < degreeBins; tIdx++)
        {
          //TODO utilizar memoria constante para senos y cosenos
          //float r = xCoord * cos(tIdx) + yCoord * sin(tIdx); //probar con esto para ver diferencia en tiempo
          float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
          int rIdx = (r + rMax) / rScale;
          //debemos usar atomic, pero que race condition hay si somos un thread por pixel? explique
          atomicAdd (acc + (rIdx * degreeBins + tIdx), 1);
        }
    }

  //TODO eventualmente cuando se tenga memoria compartida, copiar del local al global
  //utilizar operaciones atomicas para seguridad
  //faltara sincronizar los hilos del bloque en algunos lados

}

//*****************************************************************
int main (int argc, char **argv)
{
  int i;

  PGMImage inImg (argv[1]);

  vector<tuple<int, int, int, int, int>> *cpuht = new vector<tuple<int, int, int, int, int>>(degreeBins * rBins, make_tuple(0, 0, 0, 0, 0));
  int w = inImg.x_dim;
  int h = inImg.y_dim;

  //cout << "size: " << w << " x " << h << endl;

  float* d_Cos;
  float* d_Sin;

  cudaMalloc ((void **) &d_Cos, sizeof (float) * degreeBins);
  cudaMalloc ((void **) &d_Sin, sizeof (float) * degreeBins);

  //cout << "cpuht size: " << cpuht->size() << endl;
  //cout << "quick view of cpuht: " << endl;
  for (i = 0; i < 10; i++)
  {
    //cout << get<0>(cpuht->at(i)) << " " << get<1>(cpuht->at(i)) << " " << get<2>(cpuht->at(i)) << endl;
  }

  // CPU calculation
  CPU_HoughTran(inImg.pixels, w, h, cpuht);

  //////////////////////////////////////////////////// test
  vector<tuple<int, int, int, int, int>> lines; // posX0, posY0, posX1, posY1
  vector<tuple<int, int, int, int>> linesTop; // posX0, posY0, posX1, posY1

  int threshold = 50;
  float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;
  //cout << "Empieza a buscar lineas" << endl;
  for (i = 0; i < degreeBins * rBins; i++)
  {
    // cout << "vote at: " << i << ": " << get<4>(cpuht->at(i)) << endl;
    if (get<4>(cpuht->at(i)) > threshold)
    {
      //cout << "------------------------------------" << endl;
      //cout << "indice: " << i << endl;
      // convierte  theta de radianes a grados
      float theta = get<1>(cpuht->at(i)) * degreeInc * M_PI / 180;
      //cout << "theta: " << theta << endl;
      // convierte r de indice a valor real
      float r = (get<0>(cpuht->at(i)) * 2 * rMax / rBins) - rMax;
      //cout << "r: " << r << endl;
      // calcula los puntos de la linea
      // si theta es 0, entonces es una linea vertical
      if (theta == 0.0)
      {
        int personalVote = static_cast<int>(get<4>(cpuht->at(i)));
        lines.push_back(make_tuple(static_cast<int>(r)+(w/2), 0, static_cast<int>(r+(w/2)), h, personalVote));
      }
      else
      {
        // sin tomar en cuenta limites de la imagen
        float m = round(-cos(theta) / sin(theta));
        float b = r / sin(theta);

        if (m == 0.0) { // horizontal line
          float b = r / sin(theta);

          int x0 = get<2>(cpuht->at(i)) + (w/2);
          int y0 = b;
          int x1 = get<2>(cpuht->at(i)) + (w/2);
          int y1 = static_cast<int>(b);

          int personalVote = static_cast<int>(get<4>(cpuht->at(i)));
          lines.push_back(make_tuple(x0, y0, x1, y1, personalVote));

          x0 = get<2>(cpuht->at(i)) + (w/2);
          y0 = static_cast<int>(x0 + b);
          y1 = static_cast<int>(b);
          x1 = get<2>(cpuht->at(i)) + (w/2);
          

          lines.push_back(make_tuple(x0, y0, x1, y1, personalVote));


          continue;
        }
        //cout << "m: " << m << endl;
        //cout << "b: " << b << endl;
        int x0 = get<2>(cpuht->at(i)) + (w/2);
        int y0 = -get<3>(cpuht->at(i)) + (h/2);
        int x1 = w;
        int y1 = static_cast<int>((m * x1 + b));
        // con limites de la imagen
        int personalVote = static_cast<int>(get<4>(cpuht->at(i)));
        lines.push_back(make_tuple(x0, y0, x1, y1, personalVote));
        // lineas del otro lado
        // cout << " line 1 " << x0 << " " << y0 << " " << x1 << " " << y1 << endl;
        x0 = get<2>(cpuht->at(i)) + (w/2);
        y0 = -get<3>(cpuht->at(i)) + (h/2);
 
        y1 = static_cast<int>((m * x0 + b));
        cout << "pendiente negativa"<< m;
        x1 = static_cast<int>(((y1 - b)/m));
        // cout << " line 2 " << x0 << " " << y0 << " " << x1 << " " << y1 << endl;

        lines.push_back(make_tuple(x0, y0, x1, y1, personalVote));
      }
    }
  }
  /////////////////////////////////////////////////////////
  //cimg_library::CImg<unsigned char> image("./cuadrosHough.pgm");
  cimg_library::CImg<unsigned char> image(w, h, 1, 3, 255);
  
  if (image.is_empty()) {
      std::cout << "Could not open or find the image." << std::endl;
      return -1;
  }
  
  // Draw lines on the image
  //int x0 = 100, y0 = 100, x1 = 200, y1 = 200;
  const unsigned char red[] = { 255,0,0 };
  const unsigned char black[] = { 0,0,0 };

  const float opacity = 1;
  //const unsigned int pattern = ~0U;
  //image.draw_line(x0,y0,x1,y1,red,opacity);
 

  sort(lines.begin(), lines.end(), compareTuples);
  for (const auto& tuple : lines) {
      cout << "Valor: " << get<4>(tuple) << endl;
  }

  for (size_t i = 0; i < 4; i++)
  {
      cout << " line 1 " << get<0>(lines[i]) << " " << get<1>(lines[i]) << " " << get<2>(lines[i]) << " " << get<3>(lines[i]) << endl;
    image.draw_line(get<0>(lines[i]), get<1>(lines[i]), get<2>(lines[i]), get<3>(lines[i]), red, opacity);
  }
  
  // image.draw_line(42, 141, 0, -1701, black, opacity);

  // for (auto line : lines)
  // {
  //   image.draw_line(get<0>(line), get<1>(line), get<2>(line), get<3>(line), red, opacity);
  // }

  /*int j;
  for (i = 0; i < w; i++) {
    for (j = 0; j < h; j++) {
      int idx = j * w + i;
      ////cout << (int)inImg.pixels[idx] << " ";
      if (inImg.pixels[idx] >= 50) {
        image(i, j, 0) = 0;
        image(i, j, 1) = 255;
        image(i, j, 2) = 0;
      }
    }
  }*/

  // Save the modified image to a new file
  image.save("test.bmp");
  
  std::cout << "Image processing complete." << std::endl;
  //////////////////////////////////////////////////////

  // pre-compute values to be stored
  float *pcCos = (float *) malloc (sizeof (float) * degreeBins);
  float *pcSin = (float *) malloc (sizeof (float) * degreeBins);
  float rad = 0;
  for (i = 0; i < degreeBins; i++)
  {
    pcCos[i] = cos (rad);
    pcSin[i] = sin (rad);
    rad += radInc;
  }

  //float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;
  float rScale = 2 * rMax / rBins;

  // TODO eventualmente volver memoria global
  cudaMemcpy(d_Cos, pcCos, sizeof (float) * degreeBins, cudaMemcpyHostToDevice);
  cudaMemcpy(d_Sin, pcSin, sizeof (float) * degreeBins, cudaMemcpyHostToDevice);

  // setup and copy data from host to device
  unsigned char *d_in, *h_in;
  int *d_hough, *h_hough;

  h_in = inImg.pixels; // h_in contiene los pixeles de la imagen

  h_hough = (int *) malloc (degreeBins * rBins * sizeof (int));

  cudaMalloc ((void **) &d_in, sizeof (unsigned char) * w * h);
  cudaMalloc ((void **) &d_hough, sizeof (int) * degreeBins * rBins);
  cudaMemcpy (d_in, h_in, sizeof (unsigned char) * w * h, cudaMemcpyHostToDevice);
  cudaMemset (d_hough, 0, sizeof (int) * degreeBins * rBins);

  // execution configuration uses a 1-D grid of 1-D blocks, each made of 256 threads
  //1 thread por pixel
  int blockNum = ceil (w * h / 256);
  GPU_HoughTran <<< blockNum, 256 >>> (d_in, w, h, d_hough, rMax, rScale, d_Cos, d_Sin);

  // get results from device
  cudaMemcpy (h_hough, d_hough, sizeof (int) * degreeBins * rBins, cudaMemcpyDeviceToHost);

  // compare CPU and GPU results
  /*for (i = 0; i < degreeBins * rBins; i++)
  {
    if (cpuht[i] != h_hough[i])
      printf ("Calculation mismatch at : %i %i %i\n", i, cpuht[i], h_hough[i]);
  }*/
  printf("Done!\n");

  // imprimir pixeles en imagen
  PGMImage outImg (degreeBins, rBins, 1);
  outImg.pixels = (unsigned char *) h_hough;
  outImg.write(argv[2]);

  // TODO clean-up
  
  return 0;
}
