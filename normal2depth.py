"""
 ----------------------------------------------------
  @Author: tsukasa
  @Affiliation: Waseda University
  @Email: rinsa@suou.waseda.jp
  @Date: 2019-02-28 01:53:06
  @Last Modified by:   rinsa318
  @Last Modified time: 2019-04-05 17:51:54
 ----------------------------------------------------


"""


import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
import cv2
import sys
import os




def comp_depth(mask, normal):

  '''
  "arbitrary point p(x, y, Z(x, y)) and Np = (nx, ny, nz)"

  v1 = (x+1, y, Z(x+1, y)) - p
     = (1, 0, Z(x+1, y) - Z(x, y))

  Then, dot(Np, v1) == 0 #right
  0 = Np * v1
    = (nx, ny, nz) * (1, 0, Z(x+1,y) - Z(x, y))
    = nx + nz(Z(x+1,y) - Z(x, y))

  --> Z(x+1,y) - Z(x, y) = -nx/nz = p

  Also dot(Np, v2) is same #up
  --> Z(x,y+1) - Z(x, y) = -ny/nz = q

  
  Finally, apply least square to find Z(x, y).
  A: round matrix
  x: matrix of Z(x, y)
  b: matrix of p and q 

  A*x = b


  (--> might be left bottom as well???)

  '''


  ## 1. prepare matrix for least square
  print("prepare matrix for least square: Ax = b")
  A = sp.lil_matrix((mask.size * 2, mask.size))
  b = np.zeros(A.shape[0], dtype=np.float32)


  ## 2. set normal
  nx = normal[:,:,0].ravel()
  ny = normal[:,:,1].ravel()
  nz = normal[:,:,2].ravel()
  

  ## 3. fill b 
  ##  --> 0~nx.shape[0] is for v1
  ##  --> .... v2, v3, v4
  b[0:nx.shape[0]] = -nx/(nz+1e-8)
  b[nx.shape[0]:b.shape[0]] = -ny/(nz+1e-8)
  


  ## 4. fill A 
  dif= mask.size
  w = mask.shape[1]
  h = mask.shape[0]
  for i in range(mask.shape[0]):
    progress_bar(i, mask.shape[0] -1)

    for j in range(mask.shape[1]):
      # print(i)
      ## current pixel om matrix
      pixel = (i * w) + j
      

      ## for v1(right)
      if j != w-1:
        A[pixel, pixel]   = -1
        A[pixel, pixel+1] = 1

      ## for v2(up)
      if i != h-1:
        A[pixel+dif, pixel]   = -1
        A[pixel+dif, pixel+w] = 1
  


  ## 5. solve Ax = b
  print("\nsoloving Ax = b ...")
  AtA = A.transpose().dot(A)
  Atb = A.transpose().dot(b)
  x, info = sp.linalg.cg(AtA, Atb)
  


  ## 6. create output matrix
  depth = x.reshape(mask.shape)
  depth -= np.min(depth)
  depth[mask == 0] = 0.0


  return depth



def comp_depth_4edge(mask, normal):

  '''
  "arbitrary point p(x, y, Z(x, y)) and Np = (nx, ny, nz)"

  v1 = (x+1, y, Z(x+1, y)) - p
     = (1, 0, Z(x+1, y) - Z(x, y))

  Then, dot(Np, v1) == 0 #right
  0 = Np * v1
    = (nx, ny, nz) * (1, 0, Z(x+1,y) - Z(x, y))
    = nx + nz(Z(x+1,y) - Z(x, y))

  --> Z(x+1,y) - Z(x, y) = -nx/nz = p

  Also dot(Np, v2) is same #up
  --> Z(x,y+1) - Z(x, y) = -ny/nz = q

  Also dot(Np, v3) is same #left
  --> Z(x-1,y) - Z(x, y) = -nx/nz = q

  Also dot(Np, v4) is same #bottom
  --> Z(x,y-1) - Z(x, y) = -ny/nz = q
  

  Finally, apply least square to find Z(x, y).
  A: round matrix
  x: matrix of Z(x, y)
  b: matrix of p and q 

  A*x = b



  '''


  ## 1. prepare matrix for least square
  print("prepare matrix for least square: Ax = b")
  A = sp.lil_matrix((mask.size * 4, mask.size))
  b = np.zeros(A.shape[0], dtype=np.float32)


  ## 2. set normal
  nx = normal[:,:,0].ravel()
  ny = normal[:,:,1].ravel()
  nz = normal[:,:,2].ravel()


  ## 3. fill b 
  ##  --> 0~nx.shape[0] is for v1
  ##  --> .... v2, v3, v4
  b[0:nx.shape[0]]               = -nx/(nz+1e-8)
  b[nx.shape[0]:2*nx.shape[0]]   = -ny/(nz+1e-8)
  b[2*nx.shape[0]:3*nx.shape[0]] = -nx/(nz+1e-8)
  b[3*nx.shape[0]:b.shape[0]]    = -ny/(nz+1e-8)
  

  ## 4. fill A 
  dif= mask.size
  w = mask.shape[1]
  h = mask.shape[0]
  for i in range(mask.shape[0]):
    progress_bar(i, mask.shape[0] -1)

    for j in range(mask.shape[1]):

      ## current pixel om matrix
      pixel = (i * w) + j
      

      ## for v1(right)
      if j != w-1:
        A[pixel, pixel]   = -1
        A[pixel, pixel+1] = 1

      ## for v2(up)
      if i != h-1:
        A[pixel+dif, pixel]   = -1
        A[pixel+dif, pixel+w] = 1
  

      ## for v3(left)
      if j != 0:
        A[pixel+(2*dif), pixel]   = 1
        A[pixel+(2*dif), pixel-1] = -1

      ## for v4(bottom)
      if i != 0:
        A[pixel+(3*dif), pixel]   = 1
        A[pixel+(3*dif), pixel-w] = -1


  ## 5. solve Ax = b
  print("\nsoloving Ax = b ...")
  AtA = A.transpose().dot(A)
  Atb = A.transpose().dot(b)
  x, info = sp.linalg.cg(AtA, Atb)
  

  ## 6. create output matrix
  depth = x.reshape(mask.shape)
  depth -= np.min(depth)
  depth[mask == 0] = 0.0

  return depth


def progress_bar(n, N):

  '''
  print current progress
  '''

  step = 2
  percent = float(n) / float(N) * 100

  ## convert percent to bar
  current = "#" * int(percent//step)
  remain = " " * int(100/step-int(percent//step))
  bar = "|{}{}|".format(current, remain)
  print("\r{}:{:3.0f}[%]".format(bar, percent), end="", flush=True)