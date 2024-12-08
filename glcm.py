from skimage.feature import graycomatrix, graycoprops
import numpy as np

def calculate_glcm_from_single_image(
    image, distances, angles, levels, symmetric=True, normed=True
):
    return graycomatrix(
        image, distances, angles, levels, symmetric=symmetric, normed=normed
    )


def calculate_glcm_from_many_images(
    images, distances, angles, levels, symmetric=True, normed=True
):
    return [
        calculate_glcm_from_single_image(
            img, distances, angles, levels, symmetric, normed
        )
        for img in images
    ]


def calculate_glcm_matrix_for_each_category(
    categories, images, distances, angles, levels, symmetric=True, normed=True
):
    return {
        category: calculate_glcm_from_many_images(
            images[category], distances, angles, levels, symmetric, normed
        )
        for category in categories
    }

def extract_glcm_features(glcm, props):
    return np.array([graycoprops(glcm, prop).flatten() for prop in props]).flatten()


def extract_glcm_features_from_many_images(glcm_list, props):
    return [extract_glcm_features(glcm, props) for glcm in glcm_list]


def extract_glcm_features_for_each_category(categories, glcm_list, props):
    return {
        category: extract_glcm_features_from_many_images(glcm_list[category], props)
        for category in categories
    }