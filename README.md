# Detección de enfermedades en plantas con transfer learning

Proyecto final del curso de Machine Learning, UTEC 2026-2.

Integrantes: Miguel Angel, Carlos Flores y Leonardo De Casanova.

## Objetivo

Clasificar hojas de plantas como sanas o enfermas a partir de imágenes, usando una CNN preentrenada (ResNet50) con fine-tuning sobre el dataset PlantVillage.

## Dataset

PlantVillage: 54.305 imágenes de hojas de 14 especies en 38 clases (12 sanas y 26 enfermas). Se usa la partición oficial 80/20 agrupada por hoja.

- Hugging Face: `load_dataset("mohanty/PlantVillage", "color")`
- Repositorio original: https://github.com/spMohanty/PlantVillage-Dataset

## Estructura

```
notebooks/   notebook principal de Colab
results/     figuras y métricas exportadas
docs/        documento final en formato IEEE
```

## Cómo ejecutar

1. Abrir `notebooks/deteccion_enfermedades_resnet50.ipynb` en Google Colab.
2. Activar GPU en Entorno de ejecución > Cambiar tipo de entorno.
3. Elegir `MODO = "binario"` o `"multiclase"` en la celda de configuración.
4. Ejecutar todas las celdas. Si tarda mucho, bajar `FRACCION` a 0.3.

## Metodología

1. Reescalado a 224×224 y normalización con las estadísticas de ImageNet.
2. Data augmentation: giros, rotaciones, zoom, brillo y contraste.
3. ResNet50 preentrenada en ImageNet con cabeza de clasificación propia.
4. Entrenamiento en dos fases: cabeza con la base congelada y luego fine-tuning del bloque conv5.
5. Pesos de clase, early stopping, dropout y regularización L2.
6. Evaluación con matriz de confusión, precisión, recall, F1 y AUC.
