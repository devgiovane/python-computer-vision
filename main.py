import classifier
import functions


def main() -> None:
    # read images and labels with grayscale
    images_testing, labels_testing = functions.read_images("dataset/testing")
    images_training, labels_training = functions.read_images("dataset/training")
    functions.apply_filter(images_training[0])
    # apply filters
    images_testing = [functions.apply_filter(image) for image in images_testing]
    images_training = [functions.apply_filter(image) for image in images_training]
    # normalize histogram
    images_testing_processed = [functions.process_images(image) for image in images_testing]
    images_training_processed = [functions.process_images(image) for image in images_training]
    # hog extract characteristic
    hog_testing_characteristic = [functions.extraction_hog(image) for image in images_testing_processed]
    hog_training_characteristic = [functions.extraction_hog(image) for image in images_training_processed]
    # classifiers with hog
    classifier.bayes_train(hog_training_characteristic, labels_training, hog_testing_characteristic, labels_testing)
    classifier.knn_train(hog_training_characteristic, labels_training, hog_testing_characteristic, labels_testing)
    classifier.svm_train(hog_training_characteristic, labels_training, hog_testing_characteristic, labels_testing)
    # lbp extract characteristic
    lbp_testing_characteristic = [functions.extraction_lbp(image) for image in images_testing_processed]
    lbp_training_characteristic = [functions.extraction_lbp(image) for image in images_training_processed]
    # classifiers with lbp
    classifier.bayes_train(lbp_training_characteristic, labels_training, lbp_testing_characteristic, labels_testing)
    classifier.knn_train(lbp_training_characteristic, labels_training, lbp_testing_characteristic, labels_testing)
    classifier.svm_train(lbp_training_characteristic, labels_training, lbp_testing_characteristic, labels_testing)


if __name__ == '__main__':
    main()
