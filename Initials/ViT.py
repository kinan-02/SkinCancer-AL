from transformers import ViTFeatureExtractor, ViTModel


# Loading the Google's ViT pre-trained model.
def get_vit_model():
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    return model,feature_extractor