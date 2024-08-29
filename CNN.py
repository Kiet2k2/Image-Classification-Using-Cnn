import numpy as np
# hàm này để hiển thị các tấm ảnh
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
# để chọn ngẫu nhiên các tấm ảnh
import random
import os
# để load dữ liệu trong tập mẫu
from keras.datasets import cifar10
from keras.models import load_model
from matplotlib import image
# các hàm sau để xây dựng mô hình mạng neural
# thêm 1 lớp, thêm hàm kích hoạt activation
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,Dropout
from keras.layers import Dense, Activation, Flatten
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

#Tải tập dữ liệu
(X_train, y_train), (X_test,y_test) = datasets.cifar10.load_data()

def load_image(filename):
  # load tấm ảnh lên
  img = load_img(filename, grayscale=True, target_size=(32, 32))
  # chuyển về dạng vector
  img=img_to_array(img)
	# điều chỉnh kích thước thành một mảng 2D có kích thước (3, 1024)
  img=img.reshape(3,1024)
  img=img.astype('float32')
  #chuẩn hóa dữ liệu về 0-1
  img=img/255.0
  return img


#có 50000 hình ảnh đào tạo và 10000 hình ảnh thử nghiệm
X_train.shape
X_test.shape

y_train.shape
#Có 50000 giá trị tương ứng với 50000 tấm ảnh train

#y_train là một mảng 2D ,vì việc phân loại mảng 1D đủ tốt nên chúng ta chuyển đổi về mảng 1D
y_train = y_train.reshape(-1)

#y_test cũng tương tự
y_test = y_test.reshape(-1)

classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]


#vẽ một số hình ảnh để xem chúng là gì
def plot_sample(X, y, index):
    plt.figure(figsize = (15,2))
    plt.imshow(X[index])
    plt.xlabel(classes[y[index]])


plot_sample(X_train, y_train, 10)


#Chuẩn hóa hình ảnh thành một số từ 0 đến 1. Hình ảnh có 3 kênh (R, G, B) và mỗi giá trị trong kênh có thể nằm trong khoảng từ 0 đến 255.
#Do đó để chuẩn hóa trong phạm vi 0 -> 1, chúng ta cần chia nó bằng 255
#Chuẩn hóa dữ liệu đào tạo
X_train= X_train.astype('float32')
X_train = X_train / 255.0
#print(X_train)
X_test= X_test.astype('float32')
X_test = X_test / 255.0
#print(X_test)

#Xây dựng một mạng nơ-ron phức hợp để đào tạo hình ảnh
cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

#Huấn luyện mô hình cnn (mạng nơ-ron)
cnn.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])



# Đánh giá mô hình cnn (mạng nơ-ron)
history = cnn.fit(X_train,y_train, epochs=5,
                    validation_data=(X_test, y_test))
fig = plt.figure ()
plt.subplot( 2,1,1 )
plt.plot(history.history['accuracy'])
plt.plot (history.history['val_accuracy'])
plt.title ('model accuracy')
plt.ylabel ('accuracy')
plt.xlabel ('epoch')
plt.legend (['train', 'test'], loc='lower right')

plt.subplot (2,1,2)
plt.plot(history.history ['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss' )
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper right')
plt.tight_layout ()

# Hiển thị ngẫu nhiên 9 tấm hình trong tập test và dự đoán
#Hàm này để chọn ngẫu nhiên các tấm ảnh trong tập dữ liệu
import random
import os
predicted_classes=cnn.predict(X_test)
plt.rcParams['figure.figsize']=(9,9)
for i in range(9):
  plt.subplot(5,5,i+1)
  num=random.randint(0,len(X_test))
  plt.imshow(X_test[num])
  y_classes = [np.argmax(element) for element in predicted_classes]
  plt.xlabel(classes[y_classes[num]])
plt.tight_layout()


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # The CIFAR labels happen to be arrays,
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()



#hàm dùng để load hình từ bên ngoài
def load_image(filename):
  import numpy as np
  from PIL import Image
  # load tấm ảnh lên
  global label_packed
  image = Image.open(filename)
  image = image.resize((32,32))
  image = np.expand_dims(image, axis = 0)
  image = np.array(image)
  img=image.astype('float32')
  #chuẩn hóa dữ liệu về 0-1
  img=img/255.0
  return img
#Show hình và cho dự đoán
img_test = load_image("horse.jpg")
plt.figure(figsize=(4, 4))
plt.imshow(img_test[0])
ob = cnn.predict(img_test)
print('Dự đoán hình ảnh: ',classes[np.argmax(ob)])