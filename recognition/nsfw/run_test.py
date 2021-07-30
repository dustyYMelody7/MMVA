import os
import numpy as np
import tensorflow as tf
# from nsfw_detector import predict

def load_model(model_path):
    if model_path is None or not os.path.exists(model_path):
        raise ValueError("saved_model_path must be the valid directory of a saved model to load.")
    
    model = tf.keras.models.load_model(model_path)#, custom_objects={'KerasLayer': hub.KerasLayer})
    return model

def classify_nd_max(model, nd_images):
    """ Classify given a model, image array (numpy)...."""

    model_preds = model.predict(nd_images)
    # preds = np.argsort(model_preds, axis = 1).tolist()

    categories = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']

    probs = []
    model_preds = model_preds.tolist()
    for item in model_preds:
        class_ = item.index(max(item))
        probs.append(categories[class_])
    return probs

def load_images(image_paths, image_size, verbose=True):
    '''
    Function for loading images into numpy arrays for passing to model.predict
    inputs:
        image_paths: list of image paths to load
        image_size: size into which images should be resized
        verbose: show all of the image path and sizes loaded

    outputs:
        loaded_images: loaded images on which keras model can run predictions
        loaded_image_indexes: paths of images which the function is able to process

    '''
    loaded_images = []
    loaded_image_paths = []

    if isinstance(image_paths, str):
        if os.path.isdir(image_paths):
            parent = os.path.abspath(image_paths)
            image_paths = [os.path.join(parent, f) for f in os.listdir(image_paths) if os.path.isfile(os.path.join(parent, f))]
        elif os.path.isfile(image_paths):
            image_paths = [image_paths]

    for img_path in image_paths:
        try:
            if verbose:
                print(img_path, "size:", image_size)
            image = tf.keras.preprocessing.image.load_img(img_path, target_size=image_size)
            image = tf.keras.preprocessing.image.img_to_array(image)
            image /= 255
            loaded_images.append(image)
            loaded_image_paths.append(img_path)
        except Exception as ex:
            print("Image Load Failure: ", img_path, ex)
    # print(np.asarray(loaded_images[0]).shape)
    # cv2.imwrite('test.jpg', cv2.cvtColor(np.asarray(loaded_images[0]) * 255, cv2.COLOR_RGB2BGR))
    return np.asarray(loaded_images), loaded_image_paths

def classify(model, input_paths, image_dim=224):
    """ Classify given a model, input paths (could be single string), and image dimensionality...."""
    images, image_paths = load_images(input_paths, (image_dim, image_dim))
    probs = classify_nd_max(model, images)
    return dict(zip(image_paths, probs))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

with tf.device('/device:GPU:1'):
    model = load_model('nsfw_mobilenet2.224x224.h5')

with tf.device('/device:GPU:1'):

    # model = predict.load_model('nsfw_mobilenet2.224x224.h5')

    root_dir = 'test'

    file_dir = []
    for item in os.listdir(root_dir):
        if os.path.isfile(os.path.join(root_dir, item)):
            continue
        for val in os.listdir(os.path.join(root_dir, item)):
            path = os.path.join(root_dir, item, val)
            file_dir.append(path)
            if len(file_dir) >= 8:
                # Predict single image
                result = classify(model, file_dir)
                print(result)
                file_dir.clear()
