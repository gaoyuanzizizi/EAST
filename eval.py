from ModelHelper.Detection.DetectionModels.Template import EastDetectionTemplate
from ModelHelper.Detection.DetectionUtils import draw_infolder
from ModelHelper.Common.CommonUtils.HandleImage import copy_img_infolder
import os


if __name__ == '__main__':
    model_name = 'Fishnet99EastDetectionModel'
    checkpoint = 'xxxx.pth'
    test_folder = 'test'
    pred_folder = 'pred'
    if not os.path.exists(pred_folder):
        os.makedirs(pred_folder)

    template = EastDetectionTemplate()
    model = template.init_model(model_name=model_name)
    model = template.load_model(model=model, checkpoint=checkpoint)
    template.eval_model(model=model, test_folder=test_folder, pred_folder=pred_folder)
    copy_img_infolder(test_folder, pred_folder)
    draw_infolder(pred_folder)




