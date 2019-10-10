from ModelHelper.Detection.DetectionModels.Template import EastDetectionTemplate

if __name__ == '__main__':
    template = EastDetectionTemplate()
    train_folder = 'data/train'
    test_folder = 'data/test'
    output_folder = 'output'
    model_name = 'Fishnet99EastDetectionModel'
    train_batch = 4
    test_batch = 4
    # test every 10 epoch
    test_step = 10
    template.run(train_folder=train_folder, test_folder=test_folder, output_folder=output_folder,
                 model_name=model_name, train_batch=train_batch, test_batch=test_batch, test_step=test_step)