

import logging

import torch
import torch.nn as nn
from detectors import DETECTOR
from loss import LOSSFUNC
from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='xclip')
class XCLIPDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.fc_norm = nn.LayerNorm(768)
        # self.temporal_module = self.build_temporal_module(config)
        self.head = nn.Linear(768, 2)
        self.loss_func = self.build_loss(config)

    def build_backbone(self, config):
        from transformers import XCLIPVisionModel
        backbone = XCLIPVisionModel.from_pretrained(config['pretrained'])

        return backbone

    def build_temporal_module(self, config):
        return nn.LSTM(input_size=2048, hidden_size=512, num_layers=3, batch_first=True)

    def build_loss(self, config):
        # prepare the loss function
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()

        return loss_func

    def features(self, data_dict: dict) -> torch.tensor:
        # b, t, c, h, w = data_dict['image'].shape
        # frame_input = data_dict['image'].reshape(-1, c, h, w)
        # # get frame-level features
        # frame_level_features = self.backbone.features(frame_input)
        # frame_level_features = F.adaptive_avg_pool2d(frame_level_features, (1, 1)).reshape(b, t, -1)
        # # get video-level features
        # video_level_features = self.temporal_module(frame_level_features)[0][:, -1, :]

        batch_size, num_frames, num_channels, height, width = data_dict['image'].shape
        pixel_values = data_dict['image'].reshape(-1, num_channels, height, width)
        outputs = self.backbone(pixel_values, output_hidden_states=True)
        sequence_output = outputs['pooler_output'].reshape(batch_size, num_frames, -1)
        video_level_features = self.fc_norm(sequence_output.mean(1))

        return video_level_features

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.head(features)

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        loss = self.loss_func(pred, label)
        loss_dict = {'overall': loss}

        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}

        return metric_batch_dict

    def forward(self, data_dict: dict, inference=False) -> dict:
        features = self.features(data_dict)
        pred = self.classifier(features)
        prob = torch.softmax(pred, dim=1)[:, 1]
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features}


        return pred_dict
