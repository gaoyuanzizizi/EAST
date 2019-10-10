
# EAST: An Efficient and Accurate Scene Text Detector
## Introduction
This is a pytorch re-implementation of EAST: An Efficient and Accurate Scene Text Detector. It is very easy to use!
## Installation
Firstly, we need to install the ModelHelper.

pip install ModelHelper-0.1.5.tar.gz
## Data
Then we need to collect our own dataset, and the label shold like the following form.
x0,y0,x1,y1,x2,y2,x3,y3 format:

...

100,100,200,100,200,200,100,200

300,300,600,300,600,600,300,600

...

or like x0,y0,x1,y1,x2,y2,x3,y3,label format:

...

100,100,200,100,200,200,100,200,dog

300,300,600,300,600,600,300,600,cat

...

## Train
Finally we can train the model!


```python
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
```

## Fine tune
You just need to provide the checkpoint path.


```python
from ModelHelper.Detection.DetectionModels.Template import EastDetectionTemplate

if __name__ == '__main__':
    template = EastDetectionTemplate()
    train_folder = 'data/train'
    test_folder = 'data/test'
    output_folder = 'output'
    model_name = 'Fishnet99EastDetectionModel'
    train_batch = 4
    test_batch = 4
    test_step = 10
    checkpoint = 'xxxxx.pth'
    template.run(train_folder=train_folder, test_folder=test_folder, output_folder=output_folder,
                 model_name=model_name, train_batch=train_batch, test_batch=test_batch, test_step=test_step,
                 checkpoint=checkpoint)
```

## Use CPU
You just need to set use_gpu=False.


```python
from ModelHelper.Detection.DetectionModels.Template import EastDetectionTemplate

if __name__ == '__main__':
    use_gpu = False
    template = EastDetectionTemplate(use_gpu=use_gpu)
    train_folder = 'data/train'
    test_folder = 'data/test'
    output_folder = 'output'
    model_name = 'Fishnet99EastDetectionModel'
    train_batch = 4
    test_batch = 4
    test_step = 10
    template.run(train_folder=train_folder, test_folder=test_folder, output_folder=output_folder,
                 model_name=model_name, train_batch=train_batch, test_batch=test_batch, test_step=test_step)
```

## More Changes
If you want to make more changes, you can inherit the Template class.


```python
from ModelHelper.Detection.DetectionModels.Template import EastDetectionTemplate
from ModelHelper.Detection.DetectionModels.Dataset import EastDataset
from ModelHelper.Common.CommonUtils import get, get_valid
from ModelHelper.Common.CommonUtils.Wrapper import config

import torch
from torchvision import transforms


class MyTemplate(EastDetectionTemplate):
    def __init__(self, **kwargs):
        super(MyTemplate, self).__init__(**kwargs)

    def init_model(self, **kwargs):
        """
        if you want to change the init_model method write your code here or you can directly
        use the parent init_model method;
        :param kwargs:
        :return: model
        """
        #
        return super(MyTemplate, self).init_model(**kwargs)

    def init_trainloader(self, **kwargs):
        """
        if you want to change the init_trainloader method write your code here or you can
        directly use the parent init_trainloader;

        for example I don't want the data augmentation code, so I change the code as following;
        :param kwargs:
        :return: train loader
        """

        train_transforms = get('train_transforms', kwargs, None)
        if train_transforms is None:
            train_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        kwargs['transforms'] = train_transforms

        # train_detection_transforms = get('train_detection_transforms', kwargs, None)
        # if train_detection_transforms is None:
        #     random_crop_threshold = get('random_crop_threshold', kwargs, (1, 4))
        #     random_crop_size = get('random_crop_size', kwargs, 768)
        #     center_rotate_threshold = get('center_rotate_threshold', kwargs, (-30, 30))
        #     flip_type = get('flip_type', kwargs, 'Horizontal')
        #     flip_chance = get('flip_chance', kwargs, 0.5)
        #
        #     train_detection_transforms = DataAugmentation.Compose([
        #         DataAugmentation.Flip(flip_type, flip_chance),
        #         DataAugmentation.CenterRotate(center_rotate_threshold),
        #         DataAugmentation.RandomCrop(random_crop_threshold, random_crop_size)
        #     ])
        # kwargs['detection_transforms'] = train_detection_transforms
        train_folder = get_valid('train_folder', kwargs)
        train_dataset = EastDataset(folder=train_folder, **kwargs)
        train_batch = get('train_batch', kwargs, 4)
        train_worker = get('train_worker', kwargs, 8)
        drop_last = get('drop_last', kwargs, True)
        shuffle = get('shuffle', kwargs, True)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=train_batch * len(self.gpu),
                                                   num_workers=train_worker,
                                                   drop_last=drop_last,
                                                   shuffle=shuffle)
        train_data_num = len(train_dataset)
        print('Generate train data loader, train data folder: {}, train data num: {}'.format(train_folder,
                                                                                             train_data_num))
        return train_loader

    def init_testloader(self, **kwargs):
        """
        if you want to change the init_testloader method write your code here or you can
        directly use the parent init_testloader;
        :param kwargs:
        :return: test loader
        """

        return super(MyTemplate, self).init_testloader(**kwargs)

    def init_optimizer(self, **kwargs):
        """
        if you want to change the init_optimizer method write your code here or you can
        directly use the parent init_optimizer;
        :param kwargs:
        :return: optimizer
        """

        return super(MyTemplate, self).init_optimizer(**kwargs)

    def init_criterion(self, **kwargs):
        """
        if you want to change the init_criterion method write your code here or you can
        directly use the parent init_criterion;
        :param kwargs:
        :return: criterion
        """

        return super(MyTemplate, self).init_criterion(**kwargs)

    def train_model(self, **kwargs):
        """
        if you want to change the train_model method write your code here or you can
        directly use the parent train_model;
        :param kwargs:
        :return: train log and train loss
        """

        return super(MyTemplate, self).train_model(**kwargs)

    def test_model(self, **kwargs):
        """
        if you want to change the test_model method write your code here or you can
        directly use the parent test_model;
        :param kwargs:
        :return: test log and test loss
        """

        return super(MyTemplate, self).test_model(**kwargs)

    def eval_model(self, **kwargs):
        """
        if you want to change the eval_model method write your code here or you can
        directly use the parent eval_model;
        :param kwargs:
        :return: evaluation score
        """

        return super(MyTemplate, self).eval_model(**kwargs)

    def load_model(self, **kwargs):
        """
        if you want to change the load_model method write your code here or you can
        directly use the parent load_model;
        :param kwargs:
        :return: model
        """

        return super(MyTemplate, self).load_model(**kwargs)

    def save_model(self, **kwargs):
        """
        if you want to change the save_model method write your code here or you can
        directly use the parent save_model;
        :param kwargs:
        :return: checkpoint path
        """
        return super(MyTemplate, self).save_model(**kwargs)

    
    @config
    def run(self, **kwargs):
        """
        if you want to change the train logical, you need to change the run method,
         and you need to use the config wrapper;
        :param kwargs: 
        :return: None
        """
        pass

    # or you can directly use the parent run method
    # def run(self, **kwargs):
    #     super(MyTemplate, self).run(**kwargs)


if __name__ == '__main__':
    template = MyTemplate()
    train_folder = 'data/train'
    test_folder = 'data/test'
    output_folder = 'output'
    model_name = 'Fishnet99EastDetectionModel'
    train_batch = 4
    test_batch = 4
    test_step = 10
    template.run(train_folder=train_folder, test_folder=test_folder, output_folder=output_folder,
                 model_name=model_name, train_batch=train_batch, test_batch=test_batch, test_step=test_step)

```

## About future
Now I only provide Fishnet99EastDetectionModel which use Fishnet99 as backbone; in the future I will provide more kind of models.

If you have any questions contact me, my email is 1015165757@qq.com, thank you!
