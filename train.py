from absl import app, flags, logging
from absl.flags import FLAGS
import os
import shutil
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from core.yolov4 import YOLO, decode, compute_loss, decode_train
from core.dataset import Dataset
from core.config import cfg
import numpy as np
from core import utils
from core.utils import freeze_all, unfreeze_all
import time 


flags.DEFINE_string('model', 'yolov4', 'yolov4, yolov3')
flags.DEFINE_string('weights', None, 'pretrained weights')
flags.DEFINE_string('backup', './yolov4_weights', 'path for saving weights')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_integer('init_epoch', 0, 'initial epoch for training') 
flags.DEFINE_integer('max_to_keep', 3, 'maximum number of checkpoints to keep')
flags.DEFINE_integer('time_limit', -1, 'time limitation to terminate the training')
flags.DEFINE_boolean('test', 1, 'the frequency of evaluating on test data during the trainig')


def main(_argv):
    tic                                  = time.time()
    trainset                             = Dataset(FLAGS, is_training=True)
    testset                              = Dataset(FLAGS, is_training=False)
    logdir                               = "./data/log"
    isfreeze                             = False
    steps_per_epoch                      = len(trainset)
    first_stage_epochs                   = cfg.TRAIN.FISRT_STAGE_EPOCHS
    second_stage_epochs                  = cfg.TRAIN.SECOND_STAGE_EPOCHS
    warmup_steps                         = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch
    total_steps                          = (first_stage_epochs + second_stage_epochs) * steps_per_epoch
    current_step                         = int(float(FLAGS.init_epoch) / float(first_stage_epochs + second_stage_epochs) * total_steps) + 1
    global_steps                         = tf.Variable(current_step, trainable=False, dtype=tf.int64)

    input_layer                          = tf.keras.layers.Input([None, None, 3])
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    IOU_LOSS_THRESH                      = cfg.YOLO.IOU_LOSS_THRESH

    freeze_layers                        = utils.load_freeze_layer(FLAGS.model, FLAGS.tiny)

    feature_maps                         = YOLO(input_layer, NUM_CLASS, FLAGS.model, FLAGS.tiny)
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
#             if i == 0:
#                 bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
#             elif i == 1:
#                 bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
#             else:
#                 bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            bbox_tensor = decode_train(fm, tf.shape(fm)[1], NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            bbox_tensors.append(fm)
            bbox_tensors.append(bbox_tensor)

    model = tf.keras.Model(input_layer, bbox_tensors)
    model.summary()
    
    
    optimizer = getattr(tf.keras.optimizers, cfg.TRAIN.OPTIMIZER)()
    
    lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * (
                    (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
                )
    optimizer.lr.assign(lr.numpy())
    
    
    ckpt    = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, os.path.join(FLAGS.backup, 'tf_ckpts'), max_to_keep=FLAGS.max_to_keep)
    
    if FLAGS.weights == None:
        print("Training from scratch")
    else:
        if FLAGS.weights.split(".")[len(FLAGS.weights.split(".")) - 1] == "weights":
            utils.load_weights(model, FLAGS.weights, FLAGS.model, FLAGS.tiny)
    
        elif 'tf_ckpts' in FLAGS.weights:
            ckpt.restore(manager.latest_checkpoint)
        else:
            model.load_weights(FLAGS.weights)
        print('Restoring weights from: %s ... ' % FLAGS.weights)
    
    if os.path.exists(logdir): shutil.rmtree(logdir)
    train_logdir = os.path.join(logdir, 'train')
    test_logdir  = os.path.join(logdir, 'test')
    os.makedirs(train_logdir)
    os.makedirs(test_logdir)
    
    train_writer = tf.summary.create_file_writer(train_logdir)
    test_writer  = tf.summary.create_file_writer(test_logdir)

    # define training step function
    # @tf.function
    def train_step(image_data, target):
        with tf.GradientTape() as tape:
            pred_result = model(image_data, training=True)
            giou_loss = conf_loss = prob_loss = 0

            # optimizing process
            for i in range(len(freeze_layers)):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = compute_loss(pred, conv, target[i][0], target[i][1], STRIDES=STRIDES, NUM_CLASS=NUM_CLASS, IOU_LOSS_THRESH=IOU_LOSS_THRESH, i=i)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss
            try:
                tf.debugging.check_numerics(total_loss, 'checking for nan')
            except Exception as e:
                assert "Checking loss : Tensor had Inf values" in e.message
            
            regularization_loss  = [tf.cast(ls, total_loss.dtype) for ls in model.losses]
            regularization_loss  = tf.reduce_sum(tf.stack(regularization_loss))
            
            gradients = tape.gradient(total_loss + regularization_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            tf.print("=> STEP %4d/%4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
                     "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, total_steps, optimizer.lr.numpy(),
                                                               giou_loss, conf_loss,
                                                               prob_loss, total_loss))
            # update learning rate
            global_steps.assign_add(1)
            if global_steps < warmup_steps:
                lr = global_steps / warmup_steps * cfg.TRAIN.LR_INIT
            else:
                lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * (
                    (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
                )
            optimizer.lr.assign(lr.numpy())

            # writing summary data
            with train_writer.as_default():
                tf.summary.scalar("lr", optimizer.lr, step=global_steps)
                tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
                tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
                tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
                tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
            train_writer.flush()
    def test_step(image_data, target):
        with tf.GradientTape() as tape:
            pred_result = model(image_data, training=True)
            giou_loss = conf_loss = prob_loss = 0

            # optimizing process
            for i in range(len(freeze_layers)):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = compute_loss(pred, conv, target[i][0], target[i][1], STRIDES=STRIDES, NUM_CLASS=NUM_CLASS, IOU_LOSS_THRESH=IOU_LOSS_THRESH, i=i)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss
            
            tf.print("=> TEST STEP %4d   giou_loss: %4.2f   conf_loss: %4.2f   "
                     "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, giou_loss, conf_loss,
                                                               prob_loss, total_loss))
            with test_writer.as_default():
                tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
                tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
                tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
                tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
            test_writer.flush()

    time_terminate_flag = False
    for epoch in range(FLAGS.init_epoch, first_stage_epochs + second_stage_epochs):
        if epoch < first_stage_epochs:
            if not isfreeze:
                isfreeze = True
                for name in freeze_layers:
                    freeze = model.get_layer(name)
                    freeze_all(freeze)
        elif epoch >= first_stage_epochs:
            if isfreeze:
                isfreeze = False
                for name in freeze_layers:
                    freeze = model.get_layer(name)
                    unfreeze_all(freeze)
        for image_data, target in trainset:
            train_step(image_data, target)
            if FLAGS.time_limit>0 and (time.time()-tic)>=FLAGS.time_limit:
                time_terminate_flag = True
                break
        
        ckpt.step.assign_add(1)
        save_path = manager.save()
        
        model.save_weights(FLAGS.backup)
        
        
        
        if epoch % FLAGS.test == 0 and not time_terminate_flag:
            for image_data, target in testset:
                test_step(image_data, target)
                
                if FLAGS.time_limit>0 and (time.time()-tic)>=FLAGS.time_limit:
                    time_terminate_flag = True
                    break
        
        if time_terminate_flag:
            print('The training was terminated due to the time limit set by user')
            break
            
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
