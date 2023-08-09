import numpy as np
from tqdm import tqdm
import os

from pynimation.anim.animation import Animation
from pynimation.viewer.viewer import Viewer
from pynimation.common import data

#path1 = "data/animations/session-015 - random talk.fbx"

def load_fbx(path):
    fbx_file = data.getDataPath(path)

    # load the animation
    animation: Animation = Animation.load(fbx_file)

    # keep a copy of the original
    animationPost = animation.copy()

    return animation

#function to convert from fbx to expmap format
def fbx2expmap(animation):
    my_list=[]
    for i in tqdm(range(animation.shape[0])):
            sublist=[]
            
            idx1 = [0, 1, 2]       
            p = animation.globals[i][1].position            
            for k in idx1:
                sublist.append(p[k] * 1000.0)
               
                
            idx2 = [0, 1, 2]
            e=animation.globals[i][1].rotation.rotvec
            for k in idx2:
                sublist.append(e[k])            
                
            for j in range(2,24): 
                e = animation[i][j].rotation.rotvec #local
                for k in idx2:
                    sublist.append(e[k])
                    
            my_list.append(sublist)
    return my_list        
                                

def expmap_txtfile(expmap , filename):
    # Open a file in write mode
    with open("C:/Users/aghammaz/Desktop/Internship/Data/ExpMap/Session2/"+filename+'.txt', 'w') as file:
        # Iterate over each sublist in the list
        for sublist in expmap:
            # Convert the sublist to a string with values separated by commas
            line = ','.join(f'{x:.7f}' for x in sublist)
            # Write the line to the file
            file.write(line + '\n')



def main():
    pyn_data_path = "data/animations/Session2/"
    folder_path="C:/Users/aghammaz/Desktop/Internship/FBX/Data/Session2/"
    
    # Walk through the directory tree using os.walk()
    for root, dirs, files in os.walk(folder_path):
        for file in files: 

            # Get the complete file path
            file_path = os.path.join(root, file)  
            # Extract the base filename from the path
            filename = os.path.basename(file_path)
            print(filename)
            # Remove the file extension
            name_without_extension = os.path.splitext(filename)[0]
            # Extract the name between the "/" and ".fbx"
            filename = name_without_extension.split("/")[-1]

            
            animation = load_fbx(pyn_data_path+filename+".fbx")
            expmap_list = fbx2expmap(animation)
            expmap_txtfile(expmap_list , filename)

    



if __name__ == '__main__':
  main()






