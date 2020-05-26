#!/usr/bin/env python
# coding: utf-8

# # [IAPR 2020:][iapr2020] Project ‒ Special Project¶
# **Author:** Quentin Talon & Albéric de Lajarte  
# **Due date:** 28.05.2020  
# [iapr2018]: https://github.com/LTS5/iapr-2020
# 
# ## Extract datas
# We used pims : `pip install git+https://github.com/soft-matter/pims.git`, `skimage`, `os`, `numpy`, `matplotlib`, `argparse`, `pickle`, `gzip`, `sklearn`, `scipy`, `warnings`

# In[1]:


import argparse

parser = argparse.ArgumentParser(description='IAPR Special Project : Analyse a video to find it\'s equation shown by the robot')
parser.add_argument('--input', required=False)
parser.add_argument('--output', required=False, default="out.mp4")
parser.add_argument('-f')#To run flawlessly in jupyter-notebook
args = parser.parse_args()
print("--input {}".format(args.input))
print("--output {}".format(args.output))


# In[2]:


import os
import numpy as np
import matplotlib.pyplot as plt
import pims
import warnings

data_base_path = os.path.join(os.pardir, 'data')
data_base_path = os.path.join(os.pardir, 'data')
if(args.input is None):
    vid = pims.open(os.path.join(data_base_path, 'robot_parcours_1_rotated_0.mp4'))
else:
    vid = pims.open(args.input)
    warnings.filterwarnings("ignore")
plt.imshow(vid[0])
print(vid)


# ## Part 1: Still and moving image segmentation

# ### Background
# We simply calculate the median along all the images. This reduces the noise and removes the moving object.

# In[3]:


vid_stack = np.stack(vid, axis=0)
background = np.median(vid_stack, axis=0).astype(int)
if args.input is None:
    plt.figure(figsize = (20,20))
    plt.imshow(background)
    plt.show(block=False)


# ### Moving part
# We look at the red color, clean it a bit and look at his position, size and direction.  
# 

# In[4]:


#Few tests.
from skimage.morphology import binary_opening, binary_closing, disk, label
from skimage.measure import regionprops
def red(v, th):
    r = v[:,:,0] > th
    g = v[:,:,1] < th
    b = v[:,:,2] < th
    return np.logical_and(r, g, b)
col_threshold = 100#How to choose it ???
if args.input is None:
    fig, ax = plt.subplots(1,2, figsize=(20,10))

    red_arrow = red(vid[9], col_threshold)
    red_arrow_opened = binary_opening(red_arrow, selem=disk(2))

    ax[0].imshow(red_arrow)
    ax[0].set_title("Before opening dif")
    ax[1].imshow(red_arrow_opened)
    ax[1].set_title("After opening")
    plt.show(block=False)


# We had to create a home made rectangle mask

# In[5]:


from skimage.draw import polygon
def rect2mask(x0, y0, l_x, l_y, ang, shape):#ang in rad
    hyp = 0.5*np.sqrt(l_x**2+l_y**2)
    rect_ang = np.arctan(l_x/l_y) + np.pi/2
    top_right = (x0+np.cos(ang+rect_ang)*hyp, y0+np.sin(ang+rect_ang)*hyp)
    top_left = (x0+np.cos(ang+np.pi-rect_ang)*hyp, y0+np.sin(ang+np.pi-rect_ang)*hyp)
    bottom_right = (x0+np.cos(ang-rect_ang)*hyp, y0+np.sin(ang-rect_ang)*hyp)
    bottom_left = (x0+np.cos(ang+np.pi+rect_ang)*hyp, y0+np.sin(ang+np.pi+rect_ang)*hyp)
    poly_coordinates = np.asarray([top_right, top_left, bottom_left, bottom_right])
    rr, cc = polygon(poly_coordinates[:,0], poly_coordinates[:,1], shape=shape)
    mask = np.zeros(shape, dtype=bool)
    mask[rr, cc, :] = [True, True, True]
    return mask


# We create a mask for each frame and apply it on the background for each frame.  
# So we have an image of the background were we see it's value only at the position of the robot.

# In[6]:


def crop_image(img):
    mask = img>0
    if img.ndim==3:
        mask = mask.all(2)
    mask0,mask1 = mask.any(0),mask.any(1)
    return img[np.ix_(mask1,mask0)][2:-2, 2:-2]


# In[7]:


from skimage.transform import rotate
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu

# Find region of the robot
trajectory = [regionprops(label(binary_opening(red(v, col_threshold), disk(2))))[0] for v in vid]
trajectory_mask = [rect2mask(*point.centroid, point.major_axis_length, point.minor_axis_length, point.orientation, vid.frame_shape) for point in trajectory]

# Mask the part of the backgroung that is not in this region
hidden_background = [np.multiply(background,t) for t in trajectory_mask]
# Rotate the image prior to cropping
#rotated_image = [rotate(hidden, angle = -np.rad2deg(region.orientation) , preserve_range=True, center = (region.centroid[1], region.centroid[0]) ).astype(np.uint8) for hidden, region in zip(hidden_background,trajectory)]
# Crop the part of the image that is not in this region
#selected_background = [crop_image(rotated) for rotated in rotated_image]
# Use thresholding to have binary images
#segmented_background = [np.where(rgb2gray(selected) > threshold_otsu(rgb2gray(selected)), 1, 0) for selected in selected_background]
# Use morphological closing to remove noise.
#clean_background = [label(binary_closing(im, selem=disk(1))) for im in segmented_background]
# Visual check
#fig, axes = plt.subplots(1, 2, figsize = (5, 10))
#axes[0].imshow(segmented_background[16], cmap="gray")
#axes[1].imshow(clean_background[16], cmap="gray")


# In[8]:


#fig, axes = plt.subplots(6, 7, figsize = (10, 10))

#for axe, seg, clean in zip(axes.flatten(), segmented_background, clean_background) :
#    axe.imshow(clean, cmap="gray")
#    feature = np.std(clean)
#    
#    symbol = "T" if feature > 13 else "F"
#    axe.set_title(symbol + " var:{:.2} ".format(feature))
#fig.tight_layout()
#plt.show()


# ## Digits recognition

# ## Algo
# ### Trouver les boxes qui contiennent les symboles DONE
# En analysant le background.
# ### Lister les boxes qui sont couvertes par le robot DONE
# La première box sera de type **numéro**, la 2ème de type **opérateur**, etc.
# ### Analyser le contenu des boxes Albéric
# Par MLP entraitné sur MNIST avec rotation. Par d'autres types de descriptors pour les opérateurs.

# In[9]:


# Si mask area mask[bb_corners] == 1, tout couvert
def covered(mask, bbox):
    answer = False
    for r in [bbox[0], bbox[2]]:
        for c in [bbox[1], bbox[3]]:
            answer = answer or mask[r, c]
    if(answer and False):
        print(r,c)
    return answer


# Find the regions of interest

# In[26]:


from skimage import color
from skimage.filters import threshold_otsu
from skimage.morphology import binary_opening, binary_closing, binary_erosion, binary_dilation, disk, label
from skimage.measure import regionprops
import matplotlib.patches as mpatches

bk_g = color.rgb2gray(background) #background gray
bk_b = np.where(color.rgb2gray(bk_g) < threshold_otsu(bk_g), 1, 0) #background binary
bk_bb = binary_dilation(bk_b, disk(3)) #background with consolidated objects
bk_bb_label = label(bk_bb) #Labelise
bk_bb_regionprops = regionprops(bk_bb_label) #Measure the image
valid_boxes = [] #Keep the regionprops we wanted
for b in bk_bb_regionprops:
    if(b.bbox_area>50 and b.bbox_area < 700): #If the bounding box is bigger than 50 and smaller than 700, it's a sign
        valid_boxes.append(b)

if args.input is None:
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(bk_g, cmap='gray')
    for b in valid_boxes:
        minr, minc, maxr, maxc = b.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
print("We have {} valid_boxes".format(len(valid_boxes)))


# Look at the order they get selected

# In[11]:


#print(trajectory_mask[0].shape)
used_bb=dict()
for i, tm in enumerate(trajectory_mask):
    for bb in valid_boxes:
        if(covered(tm[:,:,0], bb.bbox)): #If the bounding box is covered
            if(bb not in used_bb.values()):#Add it to the list of used bounding boxes
                used_bb[i] = bb
print("We have {} used bounding_boxes".format(len(used_bb)))
if args.input is None:
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(bk_g, cmap='gray')
    for i, bb in enumerate(used_bb):
        minr, minc, maxr, maxc = used_bb[bb].bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=1)
        ax.text(maxc,maxr, i)
        ax.add_patch(rect)
    plt.show(block=False)


# Create a dict of used frame, with all the needed informations

# In[12]:


used_frame = []

for i, bb in enumerate(used_bb):
    frame = {"frame_number": bb,
             "type": "number" if(i%2==0) else "operator",  
            }
    frame["bbox"] = used_bb[bb].bbox
    frame["rgb_image"] = background[used_bb[bb].bbox[0]:used_bb[bb].bbox[2], used_bb[bb].bbox[1]:used_bb[bb].bbox[3]]
    frame["segmented"] = np.where(rgb2gray(frame["rgb_image"]) < threshold_otsu(rgb2gray(frame["rgb_image"])), 255, 0)
    frame["labeled"] = label(frame["segmented"])
    frame["region"] = regionprops(frame["labeled"], intensity_image=frame["segmented"] )
    used_frame.append(frame)


# ## Image Classifier
# ### MLP

# In[13]:


# Load classic MNIST
import gzip
from sklearn.neural_network import MLPClassifier
import pickle
if not os.path.isfile('pickle_MNIST_model.pkl'):
    def extract_data(filename, image_shape, image_number):
        with gzip.open(filename) as bytestream:
            bytestream.read(16)
            buf = bytestream.read(np.prod(image_shape) * image_number)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
            data = data.reshape(image_number, image_shape[0], image_shape[1])
        return data


    def extract_labels(filename, image_number):
        with gzip.open(filename) as bytestream:
            bytestream.read(8)
            buf = bytestream.read(1 * image_number)
            labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        return labels

    image_shape = (28, 28)
    train_set_size = 60000
    test_set_size = 10000

    data_MNIST = os.path.join(data_base_path, "MNIST")

    train_images_path = os.path.join(data_MNIST, 'train-images-idx3-ubyte.gz')
    train_labels_path = os.path.join(data_MNIST, 'train-labels-idx1-ubyte.gz')
    test_images_path = os.path.join(data_MNIST, 't10k-images-idx3-ubyte.gz')
    test_labels_path = os.path.join(data_MNIST, 't10k-labels-idx1-ubyte.gz')

    train_images = extract_data(train_images_path, image_shape, train_set_size)
    test_images = extract_data(test_images_path, image_shape, test_set_size)
    train_labels = extract_labels(train_labels_path, train_set_size)
    test_labels = extract_labels(test_labels_path, test_set_size)

    train_images = train_images[train_labels != 9]
    train_labels = train_labels[train_labels != 9]
    test_images = test_images[test_labels != 9]
    test_labels = test_labels[test_labels != 9]

    train_images_flat = train_images.reshape(train_images.shape[0], -1)
    test_images_flat = test_images.reshape(test_images.shape[0], -1)

    train_set_size = train_images.shape[0]
    test_set_size = test_images.shape[0]

# Load rotated MNIST
    rot_MNIST_test = np.loadtxt('../data/mnist_rotation_new/mnist_all_rotation_normalized_float_test.amat')
    rot_MNIST_train = np.loadtxt('../data/mnist_rotation_new/mnist_all_rotation_normalized_float_train_valid.amat')

    rot_MNIST_test = rot_MNIST_test[rot_MNIST_test[:,-1] != 9]
    rot_MNIST_train = rot_MNIST_train[rot_MNIST_train[:,-1] != 9]

# Combine dataset
    total_train_images = np.append(train_images_flat,rot_MNIST_test[:,:-1], axis=0)
    total_train_labels = np.append(train_labels, rot_MNIST_test[:,-1], axis=0)

    total_test_images = np.append(test_images_flat,rot_MNIST_train[:,:-1], axis=0)
    total_test_labels = np.append(test_labels, rot_MNIST_train[:,-1], axis=0)

# Train and save MLP
    mlp_adam = MLPClassifier(solver='adam', activation='relu', alpha=0.6, hidden_layer_sizes=(50, 30, 20),
                             verbose=True, random_state=1)
    mlp_adam.fit(total_train_images, total_train_labels)

# Save to file in the current working directory
    with open("pickle_MNIST_model.pkl", 'wb') as file:
        pickle.dump(mlp_adam, file)


# In[14]:


# Accuracy
def accuracy(model, data_test_flat, label_test):
    predicted = np.argmax(model.predict_proba(data_test_flat), 1)
    score =  sum(predicted == label_test) / np.shape(data_test_flat[:,0])
    print("Our score {} of correct answers".format(100*score))


# ### Test of MLP predictor
# Using the rotation MNIST test set

# In[15]:


import pickle
with open("pickle_MNIST_model.pkl", 'rb') as file:
    saved_model = pickle.load(file)


# In[16]:


# Test model accuracy with test dataset 
if args.input is None and False:
    rot_MNIST_test = np.loadtxt('../data/mnist_rotation_new/mnist_all_rotation_normalized_float_test.amat')
    rot_MNIST_test = rot_MNIST_test[rot_MNIST_test[:,-1] != 9]
    accuracy(saved_model, rot_MNIST_test[:,:-1], rot_MNIST_test[:,-1])
    #accuracy(saved_model, train_images_flat, train_labels)


# Using the video we have

# In[17]:


# Show result of MLP prediction
from skimage.transform import resize

for frame in used_frame:
    if frame["type"] == "number":
        image_mlp_sized = resize(frame["segmented"], (28, 28), anti_aliasing=True, preserve_range=True).astype(int)
        image_mlp = np.ravel(image_mlp_sized)
        frame["class"] = np.argmax(saved_model.predict_proba([image_mlp]), 1)[0]
        #frame["class"] = saved_model.predict_proba([image_mlp])
        print("Class of number is :{}".format(frame["class"]))
if args.input is None:
    fig, axes = plt.subplots(3, 4, figsize=(10,8))        
    for im, ax in zip(valid_boxes, axes.flatten()):
        image_mlp = background[im.bbox[0]:im.bbox[2], im.bbox[1]:im.bbox[3]]
        image_mlp = np.where(rgb2gray(image_mlp) < threshold_otsu(rgb2gray(image_mlp)), 255, 0)
        image_mlp_flat = np.ravel(resize(image_mlp, (28, 28), anti_aliasing=True, preserve_range=True).astype(int))

        ax.imshow(image_mlp)
        result = saved_model.predict_proba([image_mlp_flat])
        ax.set_title("{:0.0f}".format(np.max(result)/np.max(result[result!=np.max(result)])))
        #ax.set_title(np.argmax(result, 1))


# ### Operator predictor

# In[18]:


list_operator = []
for frame in used_frame:
    if frame["type"] == "operator":
        if (np.amax(frame["labeled"]))>1: #Why this ?
            frame["class"] = np.amax(frame["labeled"])
        else:            
            list_operator.append(frame)

# ADD FAKE MINUS SIGN !!!!
# list_operator


# In[19]:


if args.input is None:
    fig, axes = plt.subplots(1, 3, figsize=(10,8)) 
    for op, ax in zip(list_operator, axes.flatten()):
        ax.imshow(op["segmented"])
        ax.set_title("Per:{:.2}, Area:{}".format(op["region"][0].perimeter, op["region"][0].area ))
    


# In[20]:


from skimage.transform import ProjectiveTransform
from skimage.transform import swirl, rotate
import pickle
from skimage import io

if not os.path.isfile('operators_classification.pkl'):
    # Attention, file needed 
    img_operator = io.imread(os.path.join(data_base_path, 'original_operators.png'))
    img_operator = np.where(rgb2gray(img_operator) < threshold_otsu(rgb2gray(img_operator)), 255, 0)

    img_plus = [img_operator[:,0:347]]*1000
    img_moins = [img_operator[:,347*2:347*3]]*1000
    img_prod = [img_operator[:,347*4:]]*1000

    img_pmp = [img_plus, img_moins, img_prod]
    if args.input is None:
        plt.imshow(resize(img_plus[0], (28, 28), anti_aliasing=True, preserve_range=True).astype(int))
        print(img_plus[0].shape)

    rand_rot = np.random.normal(0, 15, 1000)
    rand_swirl = np.random.normal(0, 0.5, 1000)

    rand_Xmin = np.round(np.abs(np.random.normal(0, 2.5, 1000))).astype(int)+1
    rand_Xmax = np.round(np.abs(np.random.normal(0, 2.5, 1000))).astype(int)+1
    rand_Ymin = np.round(np.abs(np.random.normal(0, 2.5, 1000))).astype(int)+1
    rand_Ymax = np.round(np.abs(np.random.normal(0, 2.5, 1000))).astype(int)+1

    img_pmp_trans = [[ rotate(swirl(resize(img_op, (40,40), preserve_range=True), strength = rand_swirl[i], preserve_range=True), angle = rand_rot[i]) for i, img_op in enumerate(img_op_list)] for img_op_list in img_pmp]

    img_pmp_trans = [[ resize(img_op[rand_Xmin[i]:-rand_Xmax[i], rand_Ymin[i]:-rand_Ymax[i]], (28,28), anti_aliasing=True, preserve_range=True).astype(int) for i, img_op in enumerate(img_op_list)] for img_op_list in img_pmp_trans]

    img_pmp_trans_region = [[regionprops(label(np.where(img_op > threshold_otsu(img_op), 255, 0)), intensity_image= img_op) for img_op in img_op_list] for img_op_list in img_pmp_trans]
    operators_classification = dict()
    operators_classification["mean_feature"] = [[np.mean([img_op[0].solidity for img_op in img_op_list]), np.mean([img_op[0].eccentricity for img_op in img_op_list])] for img_op_list in img_pmp_trans_region]
    operators_classification["var_feature"] = [[np.var([img_op[0].solidity for img_op in img_op_list]), np.var([img_op[0].eccentricity for img_op in img_op_list])] for img_op_list in img_pmp_trans_region]
    operators_classification["img_pmp_trans_region"] = img_pmp_trans_region
    with open("operators_classification.pkl", 'wb') as file: # We should dump only the needed things, not the whole 42 Mo
        pickle.dump(operators_classification, file)
with open("operators_classification.pkl", 'rb') as file:
    operators_classification = pickle.load(file)
mean_feature = operators_classification["mean_feature"]
var_feature = operators_classification["var_feature"]
img_pmp_trans_region = operators_classification["img_pmp_trans_region"]


# In[21]:


from scipy.spatial.distance import mahalanobis
from numpy.linalg import inv
XX, YY = np.meshgrid(np.linspace(0.5, 1.1, 100), np.linspace(0, 1, 100))
map_feature = np.asarray([[np.argmax([-mahalanobis(np.asarray([x,y]), mean_feat, inv(np.diag(var_feat))) for mean_feat, var_feat in zip(mean_feature, var_feature)]) for x in (XX[0,:])] for y in (YY[:,0])])


# In[22]:


if args.input is None:
    fig, axe = plt.subplots(1, 1, figsize=(15,8))

    axe.pcolor(XX, YY, map_feature)

    axe.scatter([img[0].solidity for img in img_pmp_trans_region[0]], [img[0].eccentricity for img in img_pmp_trans_region[0]], marker = '+', color = 'red')
    axe.scatter([img[0].solidity for img in img_pmp_trans_region[1]], [img[0].eccentricity for img in img_pmp_trans_region[1]], marker = '^', color = 'blue')
    axe.scatter([img[0].solidity for img in img_pmp_trans_region[2]], [img[0].eccentricity for img in img_pmp_trans_region[2]], marker = 'H', color = 'green')
    axe.scatter(list_operator[0]["region"][0].solidity, list_operator[0]["region"][0].eccentricity, marker = '+', color = 'w', label = '1st operator')
    axe.scatter(list_operator[1]["region"][0].solidity, list_operator[1]["region"][0].eccentricity, marker = 'H', color = 'w', label = '2nd operator')
    axe.legend()
    axe.set_xlabel("solidity")
    axe.set_ylabel("eccentricity")


# In[23]:


import matplotlib.animation as animation
import matplotlib as mpl
def ani_frame(frames_list, positions, equation, valid_boxes, used_bb, used_frame):
    positions=np.asarray(positions)
    mpl.rcParams['savefig.pad_inches'] = 0
    fig = plt.figure( frameon=False)
    ax = plt.axes([0,0,1,1], frameon=False)
    ax.set_aspect('auto')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.autoscale(tight=True)
    fig.set_size_inches((vid.frame_shape[1]/100, vid.frame_shape[0]/100))

    for bb in valid_boxes:# Plot the valide boxes.
        minr, minc, maxr, maxc = bb.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='blue', alpha=0.5, linewidth=1)
        ax.add_patch(rect)
    time_im = ax.imshow(frames_list[0])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    time_frame_number = ax.text(10,26, 'Frame nº{:02d}'.format(0), fontsize=12, bbox=props)
    time_positions, = ax.plot([], [], ls='--', marker='o', markersize=12, color='r', lw=1, alpha=0.5, fillstyle='none')
    time_equation = ax.text(10, frames_list[0].shape[0]-20, '', fontsize=12, bbox=props)
    for frame in used_frame:
        if frame["type"] == "number":
            ax.text(frame["bbox"][3], frame["bbox"][2], frame["class"])
    def update_img(n):
        time_im.set_data(frames_list[n])
        time_positions.set_data(positions[0:n+1,1], positions[0:n+1,0])
        time_frame_number.set_text('Frame nº{:02d}'.format(n))
        time_equation.set_text('Eq: {}'.format(equation[n]))
        if n in used_bb.keys():
            minr, minc, maxr, maxc = used_bb[n].bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', alpha=1, linewidth=1)
            ax.add_patch(rect)
        return time_im
    #legend(loc=0)
    ani = animation.FuncAnimation(fig,update_img,range(len(frames_list)))
    writer = animation.writers['ffmpeg'](fps=vid.frame_rate)
    ani.save(args.output,writer=writer,dpi=100)
    return ani


# In[24]:


positions=[t.centroid for t in trajectory]
equation=['' for i in range(len(vid))]
pivot = 0
for frame in used_frame:
    if frame["type"] == "number":
        for i in np.arange(pivot, len(vid)):
            equation[i] = equation[pivot]
        equation[frame["frame_number"]] = equation[pivot]+str(frame["class"])+' '
        pivot = frame["frame_number"]
for i in np.arange(pivot, len(vid)):
    equation[i] = equation[pivot]
print(equation)


# In[25]:



ani_frame(vid, positions=positions, equation=equation, valid_boxes=valid_boxes, used_bb=used_bb, used_frame=used_frame)
vidi = pims.open(args.output)
print(vidi)


# In[ ]:




