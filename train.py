from absl import app, flags, logging
from absl.flags import FLAGS
import os
import shutil
import tensorflow as tf
import time
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# if len(physical_devices) > 0:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
strategy = tf.distribute.MirroredStrategy()

# import tensorflow_addons as tfa
from core.yolov4 import YOLO, decode, compute_loss, decode_train
from core.dataset import Dataset
from core.config import cfg
import numpy as np
from core import utils

flags.DEFINE_string('model', 'yolov4_vit_v1', 'yolov4, yolov3, yolov4_vit_v1, yolov4_vit_v2, yolov4_vit_v1_light')
flags.DEFINE_string('weights', None, 'pretrained weights')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model_path', '/kaggle', '/kaggle/')
flags.DEFINE_string('logDir', '.', '/kaggle/')
flags.DEFINE_boolean('test', False, 'include test step or not')
flags.DEFINE_integer('init_step', 0, 'initial step')
flags.DEFINE_integer('time_lim', 32000, 'time limit to terminate runtime')
flags.DEFINE_string('activation', 'gelu', 'gelu, mish')
flags.DEFINE_integer('projection_dim', 128, 'projection dim for transformer')
flags.DEFINE_integer('heads', 4, 'attention heads')
flags.DEFINE_integer('att_layer', 6, 'attention layers')
flags.DEFINE_boolean('spp', False, 'use spp layer in vit or not')
flags.DEFINE_integer('normal', 0, '0, 1, 2 for batch normalization, 3, 4 or 5 for gropu normalization')
flags.DEFINE_multi_integer('axes', [1, 2], 'axes list')

def main(_argv):
    if not os.path.isdir(FLAGS.model_path):
        raise ValueError('Path doesnt exist')

    tic = time.time()
    flg = False
    trainset = Dataset(FLAGS, is_training=True)
    testset = Dataset(FLAGS, is_training=False)
    logdir = FLAGS.logDir
    isfreeze = False
    steps_per_epoch = len(trainset)
    first_stage_epochs = cfg['TRAIN']['FISRT_STAGE_EPOCHS']
    second_stage_epochs = cfg['TRAIN']['SECOND_STAGE_EPOCHS']
    global_steps = tf.Variable(FLAGS.init_step, trainable=False, dtype=tf.int64)
    nan_counter = tf.Variable(0, trainable=False, dtype=tf.int64)
    warmup_steps = cfg['TRAIN']['WARMUP_EPOCHS'] * steps_per_epoch
    total_steps = (first_stage_epochs + second_stage_epochs) * steps_per_epoch
    # train_steps = (first_stage_epochs + second_stage_epochs) * steps_per_period
    GLOBAL_BATCH_SIZE = cfg['TRAIN']['BATCH_SIZE'] // strategy.num_replicas_in_sync
    with strategy.scope():
        input_layer = tf.keras.layers.Input([cfg['TRAIN']['INPUT_SIZE'], cfg['TRAIN']['INPUT_SIZE'], 3])
        STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
        IOU_LOSS_THRESH = cfg['YOLO']['IOU_LOSS_THRESH']
        weight_decay = 0.0005
        if FLAGS.tiny:
            num_yolo_head = 2
        else:
            num_yolo_head = 3
        feature_maps = YOLO(input_layer,
                            NUM_CLASS,
                            FLAGS.model,
                            FLAGS.tiny, 
                            FLAGS.activation, 
                            FLAGS.projection_dim,
                            [FLAGS.att_layer, FLAGS.att_layer, FLAGS.att_layer],
                            [FLAGS.heads, FLAGS.heads, FLAGS.heads],
                            FLAGS.spp,
                            FLAGS.normal,
                            FLAGS.axes)
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
        optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.TRAIN.LR_INIT)
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        model.summary()

    if FLAGS.weights == None:
        print("Training from scratch")
    else:
        model.load_weights(FLAGS.weights)
        print('Restoring weights from: %s ... ' % FLAGS.weights)


    

    if os.path.exists(logdir): shutil.rmtree(logdir)
    os.makedirs(logdir + '/train/', exist_ok = True)
    os.makedirs(logdir + '/valid/', exist_ok = True)
    writer = tf.summary.create_file_writer(logdir)
    with strategy.scope():
        # Set reduction to `none` so we can do the reduction afterwards and divide by
        # global batch size.
        def loss_(pred_result,  target):
            giou_loss = 0.0
            conf_loss = 0.0
            prob_loss = 0.0
               # optimizing process
            for i in range(num_yolo_head):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = compute_loss(pred, conv, target[i][0], target[i][1], STRIDES=STRIDES, NUM_CLASS=NUM_CLASS, IOU_LOSS_THRESH=IOU_LOSS_THRESH, i=i)
                giou_loss += tf.cast(loss_items[0], dtype = tf.float32)
                conf_loss += tf.cast(loss_items[1], dtype = tf.float32)
                prob_loss += tf.cast(loss_items[2], dtype = tf.float32)
            total_loss = giou_loss + conf_loss + prob_loss
            total_loss = tf.cast(total_loss, dtype = tf.float32)
            giou_loss  = tf.cast(giou_loss, dtype = tf.float32)
            conf_loss  = tf.cast(conf_loss, dtype = tf.float32)
            prob_loss  = tf.cast(prob_loss, dtype = tf.float32)
            
            return total_loss, giou_loss, conf_loss, prob_loss

        def computeLoss(pred_result, target):
            per_example_loss, giou_loss, conf_loss, prob_loss = loss_(pred_result, target)
            # print("kir khar per_example_loss : ", per_example_loss)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE), tf.nn.compute_average_loss(giou_loss, global_batch_size=GLOBAL_BATCH_SIZE), tf.nn.compute_average_loss(conf_loss, global_batch_size=GLOBAL_BATCH_SIZE), tf.nn.compute_average_loss(prob_loss, global_batch_size=GLOBAL_BATCH_SIZE)
    # define training step function
    # @tf.function
    def train_step(input_):
        image_data, target = input_
        with tf.GradientTape() as tape:
            pred_result = model(image_data, training=True)
            total_loss, giou_loss, conf_loss, prob_loss = computeLoss(pred_result, target)
            try:
                tf.debugging.check_numerics( total_loss, 'checking for nan')
            except Exception as e:
                assert "Checking loss : Tensor had Inf values" in e.message
            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            global_steps.assign_add(1)
            if global_steps < warmup_steps:
                lr = tf.cast(global_steps / warmup_steps * cfg.TRAIN.LR_INIT, dtype = tf.float32)
            else:
                lr =tf.cast(cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * (
                    (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
                ), dtype = tf.float32)
            optimizer.lr.assign(tf.cast(lr, tf.float32))

                # writing summary data
            with writer.as_default():
                tf.summary.scalar("train/lr", optimizer.lr, step=global_steps)
                tf.summary.scalar("train/loss/total_loss", total_loss, step=global_steps)
                tf.summary.scalar("train/loss/giou_loss", giou_loss, step=global_steps)
                tf.summary.scalar("train/loss/conf_loss", conf_loss, step=global_steps)
                tf.summary.scalar("train/loss/prob_loss", prob_loss, step=global_steps)
            writer.flush()
        return total_loss, giou_loss, conf_loss, prob_loss

    @tf.function
    def distributed_train_step(dataset_inputs):
        per_replica_losses, giou_loss, conf_loss, prob_loss = strategy.run(train_step, args=(dataset_inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None), strategy.reduce(tf.distribute.ReduceOp.SUM, giou_loss, axis=None), strategy.reduce(tf.distribute.ReduceOp.SUM, conf_loss, axis=None), strategy.reduce(tf.distribute.ReduceOp.SUM, prob_loss, axis=None)
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
    start_epoch = int((FLAGS.init_step)/total_steps*(first_stage_epochs + second_stage_epochs))
    checkpoint_prefix = os.path.join(FLAGS.model_path, "ckpt")
    for epoch in range(start_epoch, first_stage_epochs + second_stage_epochs):
        for i, (image_data, target) in enumerate(trainset) :
            toc = time.time()
            if toc - tic > FLAGS.time_lim:
                flg = True
                break
            loss, giou_loss, conf_loss, prob_loss = distributed_train_step([image_data, target])
            template = ("STEP : {}/{},  total_loss: {}, giou_loss: {}, conf_loss: {}, prob_loss: {}")
            print (template.format(global_steps.numpy(), total_steps, loss, giou_loss, conf_loss, prob_loss))
            if i % 1000 == 0 :                
                checkpoint.save(checkpoint_prefix)
                Files = os.listdir(FLAGS.model_path) 
                Files.sort()
                if len(Files) > 21 :
                    os.unlink(FLAGS.model_path + "/" + Files[1])
                    os.unlink(FLAGS.model_path + "/" + Files[2])

        if FLAGS.test:
            for image_data, target in testset:
                test_step(image_data, target)
        model.save_weights(FLAGS.model_path+'ModelF')
        print('###########################END of EPOCH{} ##############'.format(epoch))
        if flg:
            break
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass