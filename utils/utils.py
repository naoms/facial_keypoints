import matplotlib.pyplot as plt
import cv2
import numpy as np
from math import sin, cos, pi
import math


horizontal_flip = True
rotation_augmentation = True
brightness_augmentation = True
shift_augmentation = True
random_noise_augmentation = True

include_unclean_data = True    # Whether to include samples with missing keypoint values. Note that the missing values would however be filled using Pandas' 'ffill' later.
sample_image_index = 20    # Index of sample train image used for visualizing various augmentations

rotation_angles = [12]    # Rotation angle in degrees (includes both clockwise & anti-clockwise rotations)
pixel_shifts = [12]    # Horizontal & vertical shift amount in pixels (includes shift from all 4 corners)

NUM_EPOCHS = 80
BATCH_SIZE = 64



def plot_sample(image, keypoint, axis, title):
  """
  Plots your specified image alongside with its facial keypoints.
  """
  image = image.reshape(96,96)
  axis.imshow(image, cmap='gray')
  axis.scatter(keypoint[0::2], keypoint[1::2], marker='x', s=20)
  plt.title(title)

def load_images(image_data):
  """
  Extracts normalised images from the input dataframe.
  returns: list of numpy arrays with values between 0 and 1, each one corresponding to an image.
  """
  images = []
  for idx, sample in image_data.iterrows():
    image = np.array(sample['Image'].split(' '), dtype=int)
    image = np.reshape(image, (96,96,1))
    images.append(image)
  images = np.array(images)/255.
  return images

def load_keypoints(keypoint_data):
  """
  Extracts the keypoints from the input dataframe.
  returns: list of numpy arrays, each one corresponding to an image's keypoints.
  """
  keypoint_data = keypoint_data.drop('Image',axis = 1)
  keypoint_features = []
  for idx, sample_keypoints in keypoint_data.iterrows():
    keypoint_features.append(sample_keypoints)
  keypoint_features = np.array(keypoint_features, dtype = 'float')
  return keypoint_features

def left_right_flip(images, keypoints):
    flipped_keypoints = []
    flipped_images = np.flip(images, axis=2)   # Flip column-wise (axis=2)
    for idx, sample_keypoints in enumerate(keypoints):
        flipped_keypoints.append([96.-coor if idx%2==0 else coor for idx,coor in enumerate(sample_keypoints)])    # Subtract only X co-ordinates of keypoints from 96 for horizontal flipping
    return flipped_images, flipped_keypoints


from math import sin, cos, pi

def rotate_augmentation(images, keypoints):
    rotated_images = []
    rotated_keypoints = []
    print("Augmenting for angles (in degrees): ")
    for angle in rotation_angles:    # Rotation augmentation for a list of angle values
        for angle in [angle,-angle]:
            print(f'{angle}', end='  ')
            M = cv2.getRotationMatrix2D((48,48), angle, 1.0)
            angle_rad = -angle*pi/180.     # Obtain angle in radians from angle in degrees (notice negative sign for change in clockwise vs anti-clockwise directions from conventional rotation to cv2's image rotation)
            # For train_images
            for image in images:
                rotated_image = cv2.warpAffine(image, M, (96,96), flags=cv2.INTER_CUBIC)
                rotated_images.append(rotated_image)
            # For train_keypoints
            for keypoint in keypoints:
                rotated_keypoint = keypoint - 48.    # Subtract the middle value of the image dimension
                for idx in range(0,len(rotated_keypoint),2):
                    # https://in.mathworks.com/matlabcentral/answers/93554-how-can-i-rotate-a-set-of-points-in-a-plane-by-a-certain-angle-about-an-arbitrary-point
                    rotated_keypoint[idx] = rotated_keypoint[idx]*cos(angle_rad)-rotated_keypoint[idx+1]*sin(angle_rad)
                    rotated_keypoint[idx+1] = rotated_keypoint[idx]*sin(angle_rad)+rotated_keypoint[idx+1]*cos(angle_rad)
                rotated_keypoint += 48.   # Add the earlier subtracted value
                rotated_keypoints.append(rotated_keypoint)
            
    return np.reshape(rotated_images,(-1,96,96,1)), rotated_keypoints

def alter_brightness(images, keypoints):
  altered_brightness_images = []
  inc_brightness_images = np.clip(images*1.2, 0.0, 1.0)    # Increased brightness by a factor of 1.2 & clip any values outside the range of [-1,1]
  dec_brightness_images = np.clip(images*0.6, 0.0, 1.0)    # Decreased brightness by a factor of 0.6 & clip any values outside the range of [-1,1]
  altered_brightness_images.extend(inc_brightness_images)
  altered_brightness_images.extend(dec_brightness_images)
  return altered_brightness_images, np.concatenate((keypoints, keypoints))

def shift_images(images, keypoints):
  shifted_images = []
  shifted_keypoints = []
  for shift in pixel_shifts:    # Augmenting over several pixel shift values
      for (shift_x,shift_y) in [(-shift,-shift),(-shift,shift),(shift,-shift),(shift,shift)]:
          M = np.float32([[1,0,shift_x],[0,1,shift_y]])
          for image, keypoint in zip(images, keypoints):
              shifted_image = cv2.warpAffine(image, M, (96,96), flags=cv2.INTER_CUBIC)
              shifted_keypoint = np.array([(point+shift_x) if idx%2==0 else (point+shift_y) for idx, point in enumerate(keypoint)])
              if np.all(0.0<shifted_keypoint) and np.all(shifted_keypoint<96.0):
                  shifted_images.append(shifted_image.reshape(96,96,1))
                  shifted_keypoints.append(shifted_keypoint)
  shifted_keypoints = np.clip(shifted_keypoints,0.0,96.0)
  return shifted_images, shifted_keypoints

def add_noise(images):
  noisy_images = []
  for image in images:
    noisy_image = cv2.add(image, 0.008*np.random.randn(96,96,1))    # Adding random normal noise to the input image & clip the resulting noisy image between [-1,1]
    noisy_images.append(noisy_image.reshape(96,96,1))
  return noisy_images

def camera_grab(camera_id=0, fallback_filename=None):
  camera = cv2.VideoCapture(camera_id)
  try:
        # take 10 consecutive snapshots to let the camera automatically tune
        # itself and hope that the contrast and lighting of the last snapshot
        # is good enough.
      for i in range(10):
            snapshot_ok, image = camera.read()
            if snapshot_ok:
              image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      else:
          print("WARNING: could not access camera")
          if fallback_filename:
                image = plt.imread(fallback_filename)
  finally:
        camera.release()
  return image

# Function to crop the image around the face
def crop_image(image):
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        image,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(96, 96)
    )
    
    face_crop = []
    dimensions = []
    for f in faces:
        x, y, w, h = [ v for v in f ]
        cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 1)
        # Define the region of interest in the image  
        face_crop.append(image[y:y+h, x:x+w])
        face_crop_1 = face_crop[0]
        dimensions.append((x,y,w,h))
    return face_crop, dimensions

# Function to resize image
def resize_image(image):
    resized = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(resized, (96,96))
    resized = resized.tolist()
    resized = [resized]
    resized = np.array(resized)
    resized = resized.reshape(96,96,1)
    return resized


def rotate_beard(santa_filter, left_lip_coords, right_lip_coords):
  slope = (left_lip_coords[1] - right_lip_coords[1]) / (left_lip_coords[0] - right_lip_coords[0])
  angle = -math.degrees(math.atan(slope))
  return rotate_image(santa_filter,angle)


def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result


def apply_filters(image_name, model, path_to_DL, filter_name="beard"):
  """
  Takes the image name as input and applies the filter of your choice to it. 
  image_name : your image name as a string
  filter_name : can be either "beard", "glasses", or "both"
  returns: your original image with filters applied to it
  """
  image = plt.imread(image_name)
  img_copy = np.copy(image) # keep the original version
  img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2BGRA) # Used for transparency overlay of filter using the alpha channel
  cropped_images, dimensions = crop_image(image) # retrieve the faces and their coordinates
  counter=0
  santa_filter = cv2.imread(path_to_DL+'filters/santa_filter.png', -1)
  glasses = cv2.imread(path_to_DL+'filters/glasses.png', -1)

  for cropped_image in cropped_images:
    x,y,w,h = dimensions[counter]
    counter+=1
    resized_image = resize_image(cropped_image) # resizes the image to (96,96,1)
    rescaling_factor = cropped_image.shape[0]/96

    image_model = np.reshape(resized_image, (1,96,96,1))/255 # final image to feed the model
    keypoints = model.predict(image_model)[0]
    x_coords = keypoints[0::2]*rescaling_factor # coordinates for the cropped image thanks to rescaling
    y_coords = keypoints[1::2]*rescaling_factor
    left_lip_coords = (int(x_coords[11]), int(y_coords[11]))
    right_lip_coords = (int(x_coords[12]), int(y_coords[12]))
    top_lip_coords = (int(x_coords[13]), int(y_coords[13]))
    bottom_lip_coords = (int(x_coords[14]), int(y_coords[14]))
    left_eye_coords = (int(x_coords[3]), int(y_coords[3]))
    right_eye_coords = (int(x_coords[5]), int(y_coords[5]))
    brow_coords = (int(x_coords[6]), int(y_coords[6]))

    if filter_name == "beard":
      
      beard_width = left_lip_coords[0] - right_lip_coords[0]
      beard_length =  int(h/3*2) # a quarter of the face height
      shift_beard_left =  int(0.25*beard_width)
      shift_beard_top = int(0.1*beard_length)
      scale_beard_factor = 3/2

      santa_filter = cv2.resize(santa_filter, (int(beard_width*scale_beard_factor),beard_length))
      sw,sh,sc = santa_filter.shape
      santa_filter = rotate_beard(santa_filter, left_lip_coords, right_lip_coords) # align the beard to the lip angle

      # Santa filter
      for i in range(0,sw):   # Overlay the filter based on the alpha channel
        for j in range(0,sh):
          if santa_filter[i,j][3] != 0:
            try:
              img_copy[top_lip_coords[1]+y+i-shift_beard_top, right_lip_coords[0]+x+j-shift_beard_left] = santa_filter[i,j][3]
            except:
              pass
    
    if filter_name == "glasses":
      scale_percent = 50 # percent of original size
      glasses_width = int((left_eye_coords[0] - right_eye_coords[0])*3/2)
      glasses_height = int(h * scale_percent / 100)
      glasses = cv2.resize(glasses, dsize = (glasses_width, glasses_height))
      gw,gh,gc = glasses.shape
      # Glasses filter
      for i in range(0,gw):     
        for j in range(0,gh):
          if glasses[i,j][3] != 0:
            try:
              img_copy[brow_coords[1]+i+y-50, right_eye_coords[0]+j+x-50] = glasses[i,j]
            except:
              pass

    if filter_name == "both":
      # Santa
      beard_width = left_lip_coords[0] - right_lip_coords[0]
      beard_length =  int(h/3*2) # two thirds the face height
      shift_beard_left =  int(0.25*beard_width)
      shift_beard_top = int(0.1*beard_length)
      scale_beard_factor = 3/2

      # Glasses
      scale_percent = 50 # percent of original size
      glasses_width = int((left_eye_coords[0] - right_eye_coords[0])*3/2)
      glasses_height = int(h * scale_percent / 100)
      
      # Resize filters 
      santa_filter = cv2.resize(santa_filter, (int(beard_width*scale_beard_factor),beard_length))
      sw,sh,sc = santa_filter.shape
      santa_filter = rotate_beard(santa_filter, left_lip_coords, right_lip_coords) # align the beard to the lip angle
      glasses = cv2.resize(glasses, dsize = (glasses_width, glasses_height))
      gw,gh,gc = glasses.shape

      # Santa filter
      for i in range(0,sw):
        for j in range(0,sh):
          if santa_filter[i,j][3] != 0:
            try:
              img_copy[top_lip_coords[1]+y+i-shift_beard_top, right_lip_coords[0]+x+j-shift_beard_left] = santa_filter[i,j][3]
            except:
              pass
      
      # Glasses filter
      for i in range(0,gw):     
        for j in range(0,gh):
          if glasses[i,j][3] != 0:
            try:
              img_copy[brow_coords[1]+i+y-50, right_eye_coords[0]+j+x-50] = glasses[i,j]
            except:
              pass
  return img_copy
