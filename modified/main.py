import numpy as np
import cv2
from math import log, e

# Ентропія множини прикладів
def entropy(S):
    values, counts = np.unique(S, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities))

# Інформаційний виграш для ознаки A (напр. очі, ніс, рот)
def information_gain(S, A_values):
    total_entropy = entropy(S)
    unique_vals, counts = np.unique(A_values, return_counts=True)
    weighted_entropy = 0
    for val, count in zip(unique_vals, counts):
        Sv = [S[i] for i in range(len(S)) if A_values[i] == val]
        weighted_entropy += (len(Sv)/len(S)) * entropy(Sv)
    return total_entropy - weighted_entropy

# Логістична функція втрат
def logistic_loss(y, fx):
    return np.log(1 + np.exp(-y * fx))

# === ЗАВАНТАЖЕННЯ КЛАСИФІКАТОРІВ ===
face_cascade = cv2.CascadeClassifier("pattern-recognition-adaboost/cascades/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier('pattern-recognition-adaboost/cascades/haarcascade_eye_tree_eyeglasses.xml')
nose_cascade = cv2.CascadeClassifier("pattern-recognition-adaboost/cascades/haarcascade_mcs_nose.xml")
mouth_cascade = cv2.CascadeClassifier("pattern-recognition-adaboost/cascades/haarcascade_mcs_mouth.xml")

# === ОБРОБКА ЗОБРАЖЕННЯ ===
img = cv2.imread('pattern-recognition-adaboost/resources/people.jpg')
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(imgGray, 1.3, 4)

# Масив для IG
feature_vectors = []
labels = []

for (x, y, w, h) in faces:
    h = int(h + 0.2 * h)
    y = int(y - 0.1 * y)
    roi_gray = imgGray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

    # Детекція ознак
    eyes = eye_cascade.detectMultiScale(roi_gray)
    nose = nose_cascade.detectMultiScale(roi_gray, 1.3, 5)
    mouth = mouth_cascade.detectMultiScale(roi_gray, 1.7, 9)

    # Бінарні ознаки: 1 - знайдено, 0 - ні
    eyes_present = 1 if len(eyes) > 0 else 0
    nose_present = 1 if len(nose) > 0 else 0
    mouth_present = 1 if len(mouth) > 0 else 0

    feature_vectors.append([eyes_present, nose_present, mouth_present])
    labels.append(1)  # Мітка "обличчя знайдено", для прикладу

    # Малювання
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    for (nx, ny, nw, nh) in nose:
        nh = int(nh + 0.2 * nh)
        ny = int(ny - 0.2 * ny)
        nw = int(nw - 0.2 * nw)
        nx = int(nx + 0.1 * nx)
        cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (0, 255, 255), 2)
    for (mx, my, mw, mh) in mouth:
        my = int(my - 0.05 * my)
        cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (0, 0, 255), 2)

# Обчислення IG для кожної ознаки
features = np.array(feature_vectors)
labels = np.array(labels)

ig_eyes = information_gain(labels, features[:, 0])
ig_nose = information_gain(labels, features[:, 1])
ig_mouth = information_gain(labels, features[:, 2])

print(f"Information Gain - Eyes: {ig_eyes:.4f}, Nose: {ig_nose:.4f}, Mouth: {ig_mouth:.4f}")

# Демонстрація логістичної втрати на прикладі
y = 1  # позитивний клас
fx = 0.8  # передбачене значення класифікатором
loss = logistic_loss(y, fx)
print(f"Logistic loss for y={y}, f(x)={fx}: {loss:.4f}")

cv2.imshow(f'Faces detected: {len(faces)}', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
