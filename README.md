
 Digit Recognizer - Kaggle Competition

Ce projet implémente un modèle de **Deep Learning (CNN)** pour résoudre le challenge **Digit Recognizer** sur [Kaggle](https://www.kaggle.com/c/digit-recognizer).

 Description
Le dataset est basé sur **MNIST**, une base d'images manuscrites (28x28 pixels) représentant les chiffres de 0 à 9.  
L'objectif est de construire un modèle capable de reconnaître correctement ces chiffres.  

- **Train.csv** : contient 42,000 images avec leur label (0–9).  
- **Test.csv** : contient 28,000 images sans label (à prédire).  

 Méthodologie
1. Prétraitement des données :
   - Normalisation des pixels (0–255 → 0–1).
   - Reshape des images au format (28,28,1).
   - Encodage one-hot des labels.

2. Modèle :
   - CNN (Convolutional Neural Network)** avec Keras/TensorFlow.
   - Architecture :
     - Conv2D(32 filtres) + MaxPooling
     - Conv2D(64 filtres) + MaxPooling
     - Dense(128) + Dropout
     - Dense(10, softmax)

3. Entraînement :
   - Optimizer : Adam  
   - Loss : categorical_crossentropy  
   - Batch size : 128  
   - Epochs : 10  

4. Évaluation :
   - Accuracy sur validation : ~99%.  
   - Fichier **submission.csv** généré pour Kaggle.
