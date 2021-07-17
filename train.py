from absl import app, flags, logging
from absl.flags import FLAGS
import os
import shutil
import tensorflow as tf
import time
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
import tensorflow_addons as tfa
from core.yolov4 import YOLO, decode, compute_loss, decode_train
from core.dataset import Dataset
from core.config import cfg
import numpy as np
from core import utils
from core.utils import freeze_all, unfreeze_all

flags.DEFINE_string('model', 'yolov4_vit_v1', 'yolov4, yolov3, yolov4_vit_v1و yolov4_vit_v1_light')
flags.DEFINE_string('weights', None, 'pretrained weights')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model_path', '.', '/kaggle/')
flags.DEFINE_string('logDir', '.', '/kaggle/')
flags.DEFINE_boolean('test', False, 'include test step or not')
flags.DEFINE_integer('init_step', 0, 'initial step')
flags.DEFINE_integer('time_lim', 32000, 'time limit to terminate runtime')

tic = time.time()
flg = False


def main(_argv):
    if not os.path.isdir(FLAGS.model_path):
        raise ValueError('Path doesnt exist')


    trainset = Dataset(FLAGS, is_training=True)
    testset = Dataset(FLAGS, is_training=False)
    logdir = FLAGS.logDir
    isfreeze = False
    steps_per_epoch = len(trainset)
    first_stage_epochs = cfg.TRAIN.FISRT_STAGE_EPOCHS
    second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch
    total_steps = (first_stage_epochs + second_stage_epochs) * steps_per_epoch
    # train_steps = (first_stage_epochs + second_stage_epochs) * steps_per_period

    input_layer = tf.keras.layers.Input([cfg.TRAIN.INPUT_SIZE, cfg.TRAIN.INPUT_SIZE, 3])
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH
    weight_decay = 0.0005
    # freeze_layers = utils.load_freeze_layer(FLAGS.model, FLAGS.tiny)
    if FLAGS.tiny:
        num_yolo_head = 2
    else:
        num_yolo_head = 3
    feature_maps = YOLO(input_layer, NUM_CLASS, FLAGS.model, FLAGS.tiny)
    if FLAGS.tiny:
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            if i == 0:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            else:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            bbox_tensors.append(fm)
            bbox_tensors.append(bbox_tensor)
    else:
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            if i == 0:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            elif i == 1:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            else:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            bbox_tensors.append(fm)
            bbox_tensors.append(bbox_tensor)

    model = tf.keras.Model(input_layer, bbox_tensors)
    model.summary()

    if FLAGS.weights == None:
        print("Training from scratch")
    else:
        if FLAGS.weights.split(".")[len(FLAGS.weights.split(".")) - 1] == "weights":
            utils.load_weights(model, FLAGS.weights, FLAGS.model, FLAGS.tiny)
        else:
            model.load_weights(FLAGS.weights)
        print('Restoring weights from: %s ... ' % FLAGS.weights)


    #optimizer = tf.keras.optimizers.Adam()
    optimizer = tfa.optimizers.AdamW(
        learning_rate=cfg.TRAIN.LR_INIT, weight_decay=weight_decay)
    
    if os.path.exists(logdir): shutil.rmtree(logdir)
    os.makedirs(logdir + '/train/', exist_ok = True)
    os.makedirs(logdir + '/valid/', exist_ok = True)
    writer = tf.summary.create_file_writer(logdir)

    # define training step function
    # @tf.function
    def train_step(image_data, target):
        with tf.GradientTape() as tape:
            pred_result = model(image_data, training=True)
            giou_loss = conf_loss = prob_loss = 0

            # optimizing process
            for i in range(num_yolo_head):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = compute_loss(pred, conv, target[i][0], target[i][1], STRIDES=STRIDES, NUM_CLASS=NUM_CLASS, IOU_LOSS_THRESH=IOU_LOSS_THRESH, i=i)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            tf.print("=> STEP %4d/%4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
                     "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, total_steps, optimizer.lr.numpy(),
                                                               giou_loss, conf_loss,
                                                               prob_loss, total_loss))
            # update learning rate
            global_steps.assign_add(1)
            #if global_steps < warmup_steps:
            #    lr = global_steps / warmup_steps * cfg.TRAIN.LR_INIT
            #else:
            #    lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * (
            #        (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
            #    )
            #optimizer.lr.assign(lr.numpy())
            if global_steps.numpy() > 0.8 * total_steps and global_steps.numpy() < 0.9 * total_steps:
                optimizer.lr.assign(cfg.TRAIN.LR_INIT/10.0)
            elif global_steps.numpy() > 0.9 * total_steps:
                optimizer.lr.assign(cfg.TRAIN.LR_INIT/100.0)
            

            # writing summary data
            with writer.as_default():
                tf.summary.scalar("train/lr", optimizer.lr, step=global_steps)
                tf.summary.scalar("train/loss/total_loss", total_loss, step=global_steps)
                tf.summary.scalar("train/loss/giou_loss", giou_loss, step=global_steps)
                tf.summary.scalar("train/loss/conf_loss", conf_loss, step=global_steps)
                tf.summary.scalar("train/loss/prob_loss", prob_loss, step=global_steps)
            writer.flush()
    def test_step(image_data, target):
        with tf.GradientTape() as tape:
            pred_result = model(image_data, training=True)
            giou_loss = conf_loss = prob_loss = 0

            # optimizing process
            for i in range(num_yolo_head):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = compute_loss(pred, conv, target[i][0], target[i][1], STRIDES=STRIDES, NUM_CLASS=NUM_CLASS, IOU_LOSS_THRESH=IOU_LOSS_THRESH, i=i)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

            tf.print("=> TEST STEP %4d   giou_loss: %4.2f   conf_loss: %4.2f   "
                     "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, giou_loss, conf_loss,
                                                               prob_loss, total_loss))
            with writer.as_default():
                tf.summary.scalar("valid/loss/total_loss", total_loss, step=global_steps)
                tf.summary.scalar("valid/loss/giou_loss", giou_loss, step=global_steps)
                tf.summary.scalar("valid/loss/conf_loss", conf_loss, step=global_steps)
                tf.summary.scalar("valid/loss/prob_loss", prob_loss, step=global_steps)
            writer.flush()

    for epoch in range(FLAGS.init_step, first_stage_epochs + second_stage_epochs):
        # if epoch < first_stage_epochs:
        #     if not isfreeze:
        #         isfreeze = True
        #         for name in freeze_layers:
        #             freeze = model.get_layer(name)
        #             freeze_all(freeze)
        # elif epoch >= first_stage_epochs:
        #     if isfreeze:
        #         isfreeze = False
        #         for name in freeze_layers:
        #             freeze = model.get_layer(name)
        #             unfreeze_all(freeze)
        for i, (image_data, target) in enumerate(trainset) :
            toc = time.time()
            if toc - tic > FLAGS.time_lim:
                flg = True
                break
            train_step(image_data, target)
            if i % 1000 == 0 :
                #model.save(FLAGS.model_path)
                model.save_weights(FLAGS.model_path + 'ModelWeights')
        if FLAGS.test:
            for image_data, target in testset:
                test_step(image_data, target)
        model.save_weights(FLAGS.model_path)
        print('###########################END of EPOCH{} ##############'.format(epoch))
        if flg:
            break
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
