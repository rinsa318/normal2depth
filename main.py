"""
 ----------------------------------------------------
  @Author: tsukasa
  @Affiliation: Waseda University
  @Email: rinsa@suou.waseda.jp
  @Date: 2019-02-28 01:54:43
  @Last Modified by:   Tsukasa Nozawa
  @Last Modified time: 2019-02-28 23:51:38
 ----------------------------------------------------

  Usage:
   python main.py argvs[1] argvs[2] argvs[3]...
  
   argvs[1]  :  normal map parh


"""


import os
import numpy as np
import cv2
import sys
import normal2depth
import obj_functions as ob



def normalize(normal, mask):

  n = np.zeros(normal.shape)

  for i in range(normal.shape[0]):
    for j in range(normal.shape[1]):
      
      if(mask[i][j] == 0):
        continue
        
      norm = np.linalg.norm(normal[i][j])
      n[i][j] = normal[i][j] / norm

  return n



def make_albedo(depth):

  albedo = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.float32)
  albedo[:, :, 0] = 0.5
  albedo[:, :, 1] = 0.5
  albedo[:, :, 2] = 0.5


  return albedo



def mask2tiny(mask, window):

  '''
  naive approach to remove noise around border
  '''

  # mask
  mask = np.array(mask, dtype=np.uint8)
  eroded = cv2.erode(mask, np.ones((int(window), int(window)), np.uint8)) # 0~1

  return eroded



def heatmap(input):
  ''' Returns a RGB heatmap
  :input: gray scale image --> numpy array
  :return: cv2 image array 
  '''
  min = np.amin(input)
  max = np.amax(input)
  rescaled = 255 * ( (input - min ) / (max - min) )

  return cv2.applyColorMap(rescaled.astype(np.uint8), cv2.COLORMAP_JET)




def main():

  ################
  ## 1. set config
  ################
  argvs = sys.argv
  input_path = argvs[1]
  mask_path = "{0}_mask{1}".format(input_path[:-4], input_path[-4:])
  output_path = "{0}_recover.ply".format(format(input_path[:-4]))
  window_size = int(argvs[2])
  print("input path: {}".format(input_path))
  print("mask path: {}".format(mask_path))
  print("output path: {}".format(output_path))



  ################
  ## 2. load image and convert bgr to normal
  ################
  normal_image = cv2.imread(input_path)
  normal_map = ((normal_image / 255.0) * 2) - 1.0
  mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
  mask = mask2tiny(mask, window_size)
  normal_map[mask == 0] = [0.0, 0.0, 0.0]



  ################
  ## 3. (optional) flip, rotete axis
  ################
  normal = normal_map[:, :, ::-1]
  normal[:, :, 1] = normal_map.copy()[:, :, 1] * -1



  ################
  ## 4. normalize
  ################
  n = normalize(normal, mask)




  ################
  ## 5. Surface reconstruction
  ################

  ## compute depth
  depth = normal2depth.comp_depth_4edge(mask, normal)


  ## convert depth to ver and tri
  ver, tri = ob.Depth2VerTri(depth, mask)


  ## load albedo or create favarite albedo
  # cv2.imread('''albedp.png'''')
  # temp_albedo = make_albedo(depth)
  temp_albedo = heatmap(depth)/255.0

  ## output
  # 3D result
  ob.writeobj("{0}.obj".format(output_path[:-4]), ver, tri)
  ob.save_as_ply(output_path, depth, normal, temp_albedo, mask, tri)
  
  #2D result
  normal = normal.copy()[:, :, ::-1]
  normal_restore = np.array((normal+1.0)/2.0*255, dtype=np.uint8)
  depth_image = np.array((1.0 - (depth / np.max(depth))) * 255, dtype=np.uint8 )
  depth_image = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2RGB)
  results = np.hstack((normal_restore, depth_image))
  cv2.imwrite("{0}_results.png".format(output_path[:-4]), np.array(results, dtype=np.uint8))
  cv2.imshow("results", np.array(results, dtype=np.uint8))
  cv2.waitKey(0)





main()