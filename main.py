"""
 ----------------------------------------------------
  @Author: tsukasa
  @Affiliation: Waseda University
  @Email: rinsa@suou.waseda.jp
  @Date: 2019-02-28 01:54:43
 ----------------------------------------------------

"""


import os
import numpy as np
import cv2
import sys
import normal2depth
import obj_functions as ob
import argparse





def normalize(normal, mask):

  n = np.zeros(normal.shape)

  for i in range(normal.shape[0]):
    for j in range(normal.shape[1]):
      
      if(mask[i][j] == 0):
        continue
        
      norm = np.linalg.norm(normal[i][j])
      n[i][j] = normal[i][j] / norm

  return n





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





def parse_arguments():

  #### ----- set arguments
  parser = argparse.ArgumentParser(description="surface reconstruction from normal map")
  parser.add_argument("-i",
                      "--input",
                      type=str,
                      default="./example/test/test.png",
                      metavar='',
                      help='path to normal map image(OpenGL style)')
  parser.add_argument("-m",
                      "--mask",
                      type=str,
                      default=None,
                      metavar='',
                      help='path to mask image(optional)')
  args = parser.parse_args()
  args.outdir = os.path.dirname(args.input)


  #### ----- print arguments
  text  = "\n<input arguments>\n"
  for key in vars(args):
    text += "{}: {}\n".format(key.ljust(15), str(getattr(args, key)).ljust(30))
  print(text)

  return args




def main():


  ################
  ## 1. set config
  ################
  args = parse_arguments()
  output_path = "{0}_recover.ply".format(format(args.input[:-4]))



  ################
  ## 2. load image and convert bgr to normal
  ################
  normal_image = cv2.imread(args.input)
  normal_map = ((normal_image / 255.0) * 2) - 1.0
  if(args.mask is not None):
    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    # mask = mask2tiny(mask, window_size)
  else:
    mask = np.ones(normal_map[:, :, 0].shape, dtype=np.uint8) * 255
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
  # depth = normal2depth.comp_depth(mask, normal)
  depth = normal2depth.comp_depth_4edge(mask, normal)
  print("done!")


  ## convert depth to ver and tri
  ver, tri = ob.Depth2VerTri(depth, mask)


  ## load albedo or create favarite albedo
  # cv2.imread('''albedp.png'''')
  temp_albedo = normal_image
  # temp_albedo = heatmap(depth)

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