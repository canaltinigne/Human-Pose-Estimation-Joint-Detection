import matplotlib
import random
import json

def create_colors(n):
    
    cmap = matplotlib.cm.get_cmap('gist_ncar')
    colors = []

    for i in range(n):
        colors.append(cmap(random.random()))
        
    return colors

def draw_skeleton(mask_img, joint_pos, colors):
    
    skeleton = np.zeros(mask_img.shape + (3,))
    skeleton[(mask_img == 1),:] = 1. 
    
    neighbors = {
        0: [1,15,16], 1: [2,5,8], 2: [3], 3: [4], 5:[6], 6:[7], 8: [9,12], 
        9: [10], 10: [11], 11: [24,22],12: [13], 13: [14], 14: [21,19], 
        15:[17], 16:[18], 19:[20], 22:[23]
    }
    
    cl = 0
    
    for point in neighbors:
        if joint_pos[point] != (0,0):
            for neighbor in neighbors[point]:
                if joint_pos[neighbor] != (0,0):
                    cv2.line(skeleton, joint_pos[point][::-1], joint_pos[neighbor][::-1], colors[cl], 2)
                    cl += 1

    return skeleton

def openpose_joint_pos(j):
    
    x = []
    y = []
    
    joint_pos = []
    j = j.replace('.jpg', '_keypoints.json')

    for i, p in enumerate(json.load(open('SMALL_5K_OPENPOSE/' + j))['people'][0]['pose_keypoints_2d']):
        
        if i % 3 == 0:
            x.append(p)
        elif (i-1) % 3 == 0:
            y.append(p)
            
    impulses = np.zeros(cv2.imread('SMALL_5K_MASKS/' + j.replace('_keypoints.json', '.jpg')).shape[:2])
    
    y = np.floor(np.array(y)).astype('int')
    x = np.floor(np.array(x)).astype('int')
        
    for i in range(len(y)):
        if y[i] >= impulses.shape[0] or x[i] >= impulses.shape[1]:
            y[i] = 0
            x[i] = 0
            
        joint_pos.append((y[i], x[i]))
    
    return joint_pos