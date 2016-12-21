MIT License

Copyright (c) 2016 Antonio Gomez and Miguel Cardenas
(agomez@tacc.utexas.edu, miguel.cardenas@ciemat.es)
                                                                                                                                     
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.

#define pi M_PI_F
#define binsperdegree 2
#define totaldegrees 2
#define threadsperblock 4
#define min_cos 0.999390827
#define inverpi 57.29577951
#define hist_size 64
#define max_group_size 1024

#ifndef SIMD_WORK_ITEMS
#define SIMD_WORK_ITEMS 4 // default value
#endif

__kernel
__attribute((num_simd_work_items(SIMD_WORK_ITEMS)))
__attribute((work_group_size_hint(max_group_size, 1, 1)))
__attribute((max_work_group_size(max_group_size)))
void DDD_RRR (__constant double *restrict xd, __constant double *restrict yd, __constant double *restrict zd, __global int *restrict output, int numberlines, int start)
{
  int i,j,k;
  int jj,kk;

  //Make this private instead of local

  __local int local_output[max_group_size][hist_size];
  __local double local_xd[10000];
  __local double local_yd[10000];
  __local double local_zd[10000];

  if (get_global_id(0)==0) {
    for (i=0; i<numberlines; ++i) {
      local_xd[i] = xd[i];
      local_yd[i] = yd[i];
      local_zd[i] = zd[i];
    }
  }
  
  barrier (CLK_LOCAL_MEM_FENCE); 
  
  int globalelem = get_global_id(0)+start;

  #pragma unroll hist_size 
  for (i=0; i<hist_size; ++i) 
  {
    local_output[get_local_id(0)][i] = 0;
  }

  for (jj=0; jj < numberlines; jj++ ){
    double ang12, ang23, ang31;
    double acos_ang12;
    int bin12, bin23, bin31;
    ang12 = convert_float(local_xd[globalelem]*local_xd[jj] + local_yd[globalelem]*local_yd[jj] + local_zd[globalelem]*local_zd[jj]);
    ang12 = min (ang12 , 0.99999999);

#ifdef _DEBUG
    printf ("ang12 %f\n", ang12);
#endif

    if (ang12 < min_cos)
      continue;

    acos_ang12 = acos(ang12) * inverpi;
    bin12 = convert_int(acos_ang12*binsperdegree);

    for (kk=jj; kk < numberlines; kk++ ){
      ang23 = convert_double(local_xd[jj]*local_xd[kk] + local_yd[jj]*local_yd[kk] + local_zd[jj]*local_zd[kk]);
      ang23 = fmin(ang23, convert_double(0.99999999));
      if (ang23 < min_cos)
        continue;
      ang31 = convert_double(local_xd[kk]*local_xd[globalelem] + local_yd[kk]*local_yd[globalelem] + local_zd[kk]*local_zd[globalelem]);
      ang31 = fmin(ang31, convert_double(0.99999999));
	     
      if (ang31<min_cos)
        continue;

      ang23 = acos(ang23) * inverpi;
      ang31 = acos(ang31) * inverpi;
      bin23 = convert_int(ang23*binsperdegree);
      bin31 = convert_int(ang31*binsperdegree);

      local_output[get_local_id(0)][bin31*threadsperblock*threadsperblock + bin12*threadsperblock + bin23]+=1;
      if (jj!=kk){
        local_output[get_local_id(0)][bin12*threadsperblock*threadsperblock + bin31*threadsperblock + bin23]+=1;
      }
    } 
  }

  const int pos = get_local_id(0);
  #pragma unroll hist_size
  for (i=0; i<hist_size; ++i) {
    atomic_add (&output[i], local_output[pos][i]);
  }
}

