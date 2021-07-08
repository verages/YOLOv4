# -*- coding: utf-8 -*-
# @Brief: loss相关

from core.ious import box_ciou, box_iou
from nets.yolo import yolo_head
import config.config as cfg
import tensorflow as tf
from tensorflow.keras import losses


def smooth_labels(y_true, e):
    """
    u（y）表示一个关于label y，且独立于观测样本x（与x无关）的固定且已知的分布:
        q’(y|x) =（1-e) * q(y|x)+ e * u(y)

    其中，e属于[0,1]。把label y的真实分布q(y|x)与固定的分布u(y)按照1-e和e的权重混合在一起，
    构成一个新的分布。这相当于对label y中加入噪声，y值有e的概率来自于分布u(k)为方便计算，
    u(y)一般服从简单的均匀分布，则u(y)=1/K，K表示模型预测类别数目。因此，公式

        q’(y|x) = (1 - e) * q(y|x) + e/K
    :param y_true:
    :param e: [0,1]的浮点数
    :return:
    """
    k = tf.cast(tf.shape(y_true)[-1], dtype=tf.float32)
    e = tf.constant(e, dtype=tf.float32)

    return y_true * (1.0 - e) + e / k


def focal_loss(y_true, y_pred, alpha=1, gamma=2):
    """
    何凯明提出的foacl loss有助于控制正负样本的占总loss的权重、可以按照难易程度分类样本
    pt = p if y == 1 else (1 - p)
    公式FL(pt) = -α(1 - pt)^γ * log(pt)
    :param y_true:
    :param y_pred:
    :param alpha: α 范围是0 ~ 1
    :param gamma: γ
    :return:
    """
    return alpha * tf.pow(tf.abs(y_true - y_pred), gamma)


def YoloLoss(anchors, label_smooth=cfg.label_smooth):
    def compute_loss(y_true, y_pred):
        """
        计算loss
        :param y_true:
        :param y_pred:
        :return: 总的loss
        """
        # 1. 转换 y_pred -> bbox，预测置信度，各个分类的最后一层分数， 中心点坐标+宽高
        # y_pred: (batch_size, grid, grid, anchors * (x, y, w, h, obj, ...cls))
        pred_box, grid = yolo_head(y_pred, anchors, calc_loss=True)
        pred_conf = y_pred[..., 4:5]
        pred_class = y_pred[..., 5:]

        true_conf = y_true[..., 4:5]
        true_class = y_true[..., 5:]

        if label_smooth:
            true_class = smooth_labels(true_class, label_smooth)

        # 乘上一个比例，让小框的在total loss中有更大的占比，这个系数是个超参数，如果小物体太多，可以适当调大
        box_loss_scale = 2 - y_true[..., 2:3] * y_true[..., 3:4]

        # 找到负样本群组，第一步是创建一个数组，[]
        ignore_mask = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True)
        true_conf_bool = tf.cast(true_conf, tf.bool)

        # 对每一张图片计算ignore_mask
        def loop_body(b, ignore_mask):
            # true_conf_bool中，为True的值，y_true[l][b, ..., 0:4]才有效
            # 最后计算除true_box的shape[box_num, 4]
            true_box = tf.boolean_mask(y_true[b, ..., 0:4], true_conf_bool[b, ..., 0])
            # 计算预测框 和 真实框（归一化后的xywh在图中的比例）的交并比
            iou = box_iou(pred_box[b], true_box)

            # 计算每个true_box对应的预测的iou最大的box
            best_iou = tf.reduce_max(iou, axis=-1)
            # 计算出来的iou如果大于阈值则不被输入到loss计算中去，这个方法可以平衡正负样本
            ignore_mask = ignore_mask.write(b, tf.cast(best_iou < cfg.ignore_thresh, tf.float32))
            return b + 1, ignore_mask

        batch_size = tf.shape(y_pred)[0]

        # while_loop创建一个tensorflow的循环体，args:1、循环条件（b小于batch_size） 2、循环体 3、传入初始参数
        # lambda b,*args: b<m：是条件函数  b,*args是形参，b<bs是返回的结果
        _, ignore_mask = tf.while_loop(lambda b, ignore_mask: b < batch_size, loop_body, [0, ignore_mask])

        # 将每幅图的内容压缩，进行处理
        ignore_mask = ignore_mask.stack()
        ignore_mask = tf.expand_dims(ignore_mask, -1)  # 扩展维度用来后续计算loss (b,13,13,3,1,1)

        # 计算ciou损失
        raw_true_box = y_true[..., 0:4]
        ciou = box_ciou(pred_box, raw_true_box)
        ciou_loss = true_conf * box_loss_scale * (1 - ciou)

        # 如果该位置本来有框，那么计算1与置信度的交叉熵
        # 如果该位置本来没有框，而且满足best_iou<ignore_thresh，则被认定为负样本
        # best_iou<ignore_thresh用于限制负样本数量

        conf_loss = tf.nn.sigmoid_cross_entropy_with_logits(true_conf, pred_conf)
        respond_bbox = true_conf
        respond_bgd = (1 - true_conf) * ignore_mask
        # 计算focal loss
        conf_focal = focal_loss(true_conf, pred_conf)
        confidence_loss = conf_focal * (respond_bbox * conf_loss + respond_bgd * conf_loss)

        # 预测类别损失
        class_loss = true_conf * tf.nn.sigmoid_cross_entropy_with_logits(true_class, pred_class)

        # 各个损失求平均
        location_loss = tf.reduce_sum(ciou_loss) / tf.cast(batch_size, tf.float32)
        confidence_loss = tf.reduce_sum(confidence_loss) / tf.cast(batch_size, tf.float32)
        class_loss = tf.reduce_sum(class_loss) / tf.cast(batch_size, tf.float32)

        return location_loss + confidence_loss + class_loss
    return compute_loss
