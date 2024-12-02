from PIL import Image
import os
import numpy as np

def _preprocess(image_arr, th=0.4):
    image_arr[image_arr > th] = 1.0
    image_arr[image_arr <= th] = 0.0
    return image_arr



def image_resize(path):
    # path : the image path you want to resize
    # img_resized: img have resize to 360*360
    #open picture
    img = Image.open(path)
    # resize to 360*360
    img_resized = img.resize((180, 180), Image.LANCZOS)
    return img_resized

def read_images_from_folder(folder_path):
    # read images from 'folder_path'
    # flat the image and push into 'images' array
    print(f'reading from {folder_path}...')
    images = []
    length = len(os.listdir(folder_path))
    i = 0
    for filename in os.listdir(folder_path):
        i = i + 1
        progress = ((i + 1)/length)*100
        if i % 500 == 0:
            print(f"progress: {progress} %")
        if filename.lower().endswith('.png'):
            file_path = os.path.join(folder_path, filename)
            try:
                # 打開圖片
                with Image.open(file_path) as img:
                    # 將圖片轉為灰階
                    img = img.convert('L')
                    # 檢查圖片尺寸是否為 360x360
                    if img.size != (180, 180):
                        img = image_resize(file_path)
                        img.save(file_path)
                        print(file_path)
                        print(img.size)
                    if img.size == (180, 180):
                        # 將圖片轉換為 np.array 並進行標準化 (值範圍從 0 到 1)
                        img_array = np.array(img) / 255.0
                        # 將圖片展平 (flatten)
                        #img_array = img_array.flatten()
                        #print(img_array)
                        images.append(img_array)
                        #print(images[0])
                        #print(f'image size: {len(images[0])}')
                    else:
                        print(f"圖片尺寸為 {img.size}, 不是 180x180: {filename}")
            # except UnidentifiedImageError:
            #     print(f"無法識別圖片文件: {filename}")
            except Exception as e:
                print(f"無法讀取圖片: {filename}, 錯誤: {e}")
    print(f"finish reading from {folder_path}")
    return images

def preprocess_image():
    print("begin")
    preprocess_folder_test = "\photo"

    preprocess_test = np.array(read_images_from_folder(preprocess_folder_test), dtype="object")
    print(preprocess_test.shape)

    test_labels = np.ones((preprocess_test.shape[0], 1))
    print(test_labels)

    preprocess_photo = np.concatenate((preprocess_fake, preprocess_real), axis=0)
    print(preprocess_photo.shape)

    labels = np.concatenate((fake_labels, real_labels), axis=0)
    print(labels.shape)

    for i in range(len(preprocess_photo)):
        preprocess_photo[i] = _preprocess(preprocess_photo[i])
    return preprocess_photo, labels

print("begin")
preprocess_folder_test = "C:/Users/user/Desktop/photo"

preprocess_test = np.array(read_images_from_folder(preprocess_folder_test), dtype="object")
print(preprocess_test.shape)

test_labels = np.ones((preprocess_test.shape[0], 1))
test_labels[1] = 0
test_labels[2] = 0

print(test_labels)

