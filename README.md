# Animal Diet Classifier with CLIP

Given a photo of an animal, this model predicts its diet category: **Herbivore, Carnivore, Omnivore or Insectivore**. Instead of training a CNN from scratch, it uses OpenAI's CLIP (ViT-B/32) as a frozen feature extractor and trains a simple logistic regression classifier on top of the image embeddings.

## Dataset

[Animal Image Dataset (90 Different Animals)](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals) on Kaggle. I mapped 57 of the species to a diet category, giving 3,420 images:

| Category | Species | Images |
| --- | ---: | ---: |
| Herbivore | 23 | 1,380 |
| Carnivore | 17 | 1,020 |
| Omnivore | 14 | 840 |
| Insectivore | 3 | 180 |

Split 80/20 into train and test sets, stratified by category.

## Approach

1. Load CLIP ViT-B/32 and use its preprocessing pipeline on every image.
2. Encode each image into a 512-dimensional CLIP embedding (no fine-tuning).
3. Train a logistic regression classifier on the training embeddings.
4. Evaluate on the held-out test set, inspect correct and incorrect predictions, and try the model on new uploaded photos.

## Results

**Test accuracy: 97.8%** (684 test images)

| Class | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Carnivore | 0.99 | 0.97 | 0.98 |
| Herbivore | 0.96 | 0.99 | 0.98 |
| Insectivore | 1.00 | 0.97 | 0.99 |
| Omnivore | 0.98 | 0.96 | 0.97 |

Most errors were omnivores or carnivores predicted as herbivores. Because the classifier learns from what animals look like rather than what they eat, it is effectively recognising species and mapping them to a diet, so it can only be as good as that species-to-diet mapping.

## Running it

The notebook is designed for **Google Colab** with a GPU runtime.

1. Get a Kaggle API token from your Kaggle account settings.
2. In Colab, click the key icon in the left sidebar, add a secret called `KAGGLE_API_TOKEN` with your token as the value, and turn on notebook access.
3. Update the `%cd` path near the top to a folder in your own Google Drive.
4. Run all cells. The dataset downloads and unzips automatically.

The last section lets you upload your own animal photos to test the model.
