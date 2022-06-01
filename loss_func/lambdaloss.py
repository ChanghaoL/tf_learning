# coding: utf-8

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

"""
How to use
        ndcg, grads = calc_lambda_loss_gradient_v3(queries, labels_ctr, prediction_ctr, real_batch,features,self.params['lambda_loss_rate'])
        train_ops.append(self.params[optimizer].apply_gradients(map(lambda x: (grads2[x], x), train_vars[optimizer])))
"""
# 这一版代码修改了lambda_loss2，所以单独用一个文件
def calc_lambda_loss_gradient(queries, labels, prediction, batch_size):
    with tf.variable_scope('lambda_loss'):
        q_ids, idx = tf.unique(tf.squeeze(queries))
        batch_labels = tf.dynamic_partition(labels, idx, batch_size)
        batch_predictions = tf.dynamic_partition(prediction, idx, batch_size)

        losses_results = [lambda_loss2(_labels, _predictions)
                          for _labels, _predictions in zip(batch_labels, batch_predictions)]
        ndcg = tf.concat([tf.reshape(losses_results[i][0],[1]) for i in range(batch_size)],0)
        ndcg = tf.gather_nd(ndcg,tf.where(tf.logical_not(tf.is_nan(ndcg))))
    lambdas_i = tf.concat([losses_results[i][2] for i in range(batch_size)], axis=0)
    lambdas_i = tf.where(tf.logical_not(tf.is_nan(lambdas_i)), lambdas_i, tf.zeros_like(lambdas_i))
    lambdas_i = lambdas_i / tf.maximum(1.0, tf.reduce_sum(tf.abs(lambdas_i)))
    grads = tf.gradients(prediction, tf.trainable_variables(), grad_ys=tf.expand_dims(lambdas_i, 1))
    return ndcg, grads

# 直接利用距离-评分打标,且加上NDCG,并且只需要计算一次ndcg
def calc_multi_loss_gradient_v2(queries, labels, prediction, batch_size, features):
    with tf.variable_scope('lambda_loss'):
        q_ids, idx = tf.unique(tf.squeeze(queries))
        batch_labels = tf.dynamic_partition(labels, idx, batch_size)
        batch_predictions = tf.dynamic_partition(prediction, idx, batch_size)

        # lambda-loss部分
        losses_results = [lambda_loss2(_labels, _predictions)
                          for _labels, _predictions in zip(batch_labels, batch_predictions)]
        ndcg = tf.concat([tf.reshape(losses_results[i][0], [1]) for i in range(batch_size)], 0)
        ndcg = tf.gather_nd(ndcg, tf.where(tf.logical_not(tf.is_nan(ndcg))))
        # 保存的ndcg权重
        pair_weights = [losses_results[i][3] for i in range(batch_size)]

        # 这里加上利用距离-评分打标部分,暂时先不使用NDCG
        batch_scores = tf.dynamic_partition(features['avgScore'], idx, batch_size)
        batch_dis = tf.dynamic_partition(features['distance'], idx, batch_size)

        ## 距离上的
        losses_results2 = [lambda_loss_for_distance(_labels, _predictions, _weight)
                           for _labels, _predictions, _weight in zip(batch_dis, batch_predictions, pair_weights)]
        ## 评分上的
        losses_results3 = [lambda_loss_for_score(_labels, _predictions, _weight)
                           for _labels, _predictions, _weight in zip(batch_scores, batch_predictions, pair_weights)]

    lambdas_i = tf.concat([losses_results[i][2] for i in range(batch_size)], axis=0)
    lambdas_i = tf.where(tf.logical_not(tf.is_nan(lambdas_i)), lambdas_i, tf.zeros_like(lambdas_i))
    lambdas_i = lambdas_i / tf.maximum(1.0, tf.reduce_sum(tf.abs(lambdas_i)))

    lambdas_i2 = tf.concat(losses_results2, axis=0)
    lambdas_i2 = tf.where(tf.logical_not(tf.is_nan(lambdas_i2)), lambdas_i2, tf.zeros_like(lambdas_i2))
    lambdas_i2 = lambdas_i2 / tf.maximum(1.0, tf.reduce_sum(tf.abs(lambdas_i2)))

    lambdas_i3 = tf.concat(losses_results3, axis=0)
    lambdas_i3 = tf.where(tf.logical_not(tf.is_nan(lambdas_i3)), lambdas_i3, tf.zeros_like(lambdas_i3))
    lambdas_i3 = lambdas_i3 / tf.maximum(1.0, tf.reduce_sum(tf.abs(lambdas_i3)))

    tf.summary.histogram('l1', lambdas_i)
    tf.summary.histogram('l2', lambdas_i2)
    tf.summary.histogram('l3', lambdas_i3)

    # 调整权重
    rate = 0.98
    lambdas_final = rate * lambdas_i + (1 - rate) / 2 * (lambdas_i2 + lambdas_i3)

    grads = tf.gradients(prediction, tf.trainable_variables(), grad_ys=tf.expand_dims(lambdas_final, 1))
    return ndcg, grads

def calc_multi_loss_gradient_v2_1(queries, labels, prediction, batch_size, features):
    with tf.variable_scope('lambda_loss'):
        q_ids, idx = tf.unique(tf.squeeze(queries))
        batch_labels = tf.dynamic_partition(labels, idx, batch_size)
        batch_predictions = tf.dynamic_partition(prediction, idx, batch_size)

        # lambda-loss部分
        losses_results = [lambda_loss2(_labels, _predictions)
                          for _labels, _predictions in zip(batch_labels, batch_predictions)]
        ndcg = tf.concat([tf.reshape(losses_results[i][0], [1]) for i in range(batch_size)], 0)
        ndcg = tf.gather_nd(ndcg, tf.where(tf.logical_not(tf.is_nan(ndcg))))
        # 保存的ndcg权重
        pair_weights = [losses_results[i][3] for i in range(batch_size)]

        # 这里加上利用距离-评分打标部分,暂时先不使用NDCG
        batch_scores = tf.dynamic_partition(features['manualXgbFea'], idx, batch_size)
        # batch_dis = tf.dynamic_partition(features['distance'], idx, batch_size)

        # ## 距离上的
        # losses_results2 = [lambda_loss_for_distance(_labels, _predictions, _weight)
        #                    for _labels, _predictions, _weight in zip(batch_dis, batch_predictions, pair_weights)]
        ## 评分上的
        losses_results3 = [lambda_loss_for_score(_labels, _predictions, _weight)
                           for _labels, _predictions, _weight in zip(batch_scores, batch_predictions, pair_weights)]

    lambdas_i = tf.concat([losses_results[i][2] for i in range(batch_size)], axis=0)
    lambdas_i = tf.where(tf.logical_not(tf.is_nan(lambdas_i)), lambdas_i, tf.zeros_like(lambdas_i))
    lambdas_i = lambdas_i / tf.maximum(1.0, tf.reduce_sum(tf.abs(lambdas_i)))

    # lambdas_i2 = tf.concat(losses_results2, axis=0)
    # lambdas_i2 = tf.where(tf.logical_not(tf.is_nan(lambdas_i2)), lambdas_i2, tf.zeros_like(lambdas_i2))
    # lambdas_i2 = lambdas_i2 / tf.maximum(1.0, tf.reduce_sum(tf.abs(lambdas_i2)))

    lambdas_i3 = tf.concat(losses_results3, axis=0)
    lambdas_i3 = tf.where(tf.logical_not(tf.is_nan(lambdas_i3)), lambdas_i3, tf.zeros_like(lambdas_i3))
    lambdas_i3 = lambdas_i3 / tf.maximum(1.0, tf.reduce_sum(tf.abs(lambdas_i3)))

    tf.summary.histogram('l1', lambdas_i)
    # tf.summary.histogram('l2', lambdas_i2)
    tf.summary.histogram('l3', lambdas_i3)

    # 调整权重
    rate = 0.98
    lambdas_final = rate * lambdas_i + (1 - rate) * lambdas_i3

    grads = tf.gradients(prediction, tf.trainable_variables(), grad_ys=tf.expand_dims(lambdas_final, 1))
    return ndcg, grads

# 直接利用距离-评分打标,且加上NDCG,并且只需要计算一次ndcg
# 简单的加上其他感知信息,目前共加到6种
def calc_multi_loss_gradient_v3(queries, labels, prediction, batch_size, features):
    with tf.variable_scope('lambda_loss'):
        q_ids, idx = tf.unique(tf.squeeze(queries))
        batch_labels = tf.dynamic_partition(labels, idx, batch_size)
        batch_predictions = tf.dynamic_partition(prediction, idx, batch_size)

        # lambda-loss部分
        losses_results = [lambda_loss2(_labels, _predictions)
                          for _labels, _predictions in zip(batch_labels, batch_predictions)]
        ndcg = tf.concat([tf.reshape(losses_results[i][0], [1]) for i in range(batch_size)], 0)
        ndcg = tf.gather_nd(ndcg, tf.where(tf.logical_not(tf.is_nan(ndcg))))
        # 保存的ndcg权重
        pair_weights = [losses_results[i][3] for i in range(batch_size)]

        # 这里加上利用距离-评分打标部分,暂时先不使用NDCG
        batch_scores = tf.dynamic_partition(features['avgScore'], idx, batch_size)
        batch_dis = tf.dynamic_partition(features['distance'], idx, batch_size)

        batch_rel = tf.dynamic_partition(features['querypoiNameSim'], idx, batch_size)
        batch_close = tf.dynamic_partition(features['closeStatus'], idx, batch_size)
        batch_online = tf.dynamic_partition(features['nowOnline'], idx, batch_size)
        batch_pic = tf.dynamic_partition(features['isRevealPic'], idx, batch_size)

        ## 距离上的
        losses_results2 = [lambda_loss_for_distance(_labels, _predictions, _weight)
                           for _labels, _predictions, _weight in zip(batch_dis, batch_predictions, pair_weights)]
        ## 评分上的
        losses_results3 = [lambda_loss_for_score(_labels, _predictions, _weight)
                           for _labels, _predictions, _weight in zip(batch_scores, batch_predictions, pair_weights)]
        ## bm25上的,相关性
        losses_results4 = [lambda_loss_for_score(_labels, _predictions, _weight)
                           for _labels, _predictions, _weight in zip(batch_rel, batch_predictions, pair_weights)]
        ## 歇业,0是正常
        losses_results5 = [lambda_loss_for_sparse(_labels, _predictions, _weight, 0)
                           for _labels, _predictions, _weight in zip(batch_close, batch_predictions, pair_weights)]
        ## 营业,1是正常
        losses_results6 = [lambda_loss_for_sparse(_labels, _predictions, _weight, 1)
                           for _labels, _predictions, _weight in zip(batch_online, batch_predictions, pair_weights)]
        ## 头图,0是正常(1应该是采用兜底图)
        losses_results7 = [lambda_loss_for_sparse(_labels, _predictions, _weight, 0)
                           for _labels, _predictions, _weight in zip(batch_pic, batch_predictions, pair_weights)]

    lambdas_i = tf.concat([losses_results[i][2] for i in range(batch_size)], axis=0)
    lambdas_i = tf.where(tf.logical_not(tf.is_nan(lambdas_i)), lambdas_i, tf.zeros_like(lambdas_i))
    lambdas_i = lambdas_i / tf.maximum(1.0, tf.reduce_sum(tf.abs(lambdas_i)))

    lambdas_i2 = tmp_post_lambdas(losses_results2)
    lambdas_i3 = tmp_post_lambdas(losses_results3)
    lambdas_i4 = tmp_post_lambdas(losses_results4)
    lambdas_i5 = tmp_post_lambdas(losses_results5)
    lambdas_i6 = tmp_post_lambdas(losses_results6)
    lambdas_i7 = tmp_post_lambdas(losses_results7)

    tf.summary.histogram('l1', lambdas_i)
    tf.summary.histogram('l2', lambdas_i2)
    tf.summary.histogram('l3', lambdas_i3)
    tf.summary.histogram('l4', lambdas_i4)
    tf.summary.histogram('l5', lambdas_i5)
    tf.summary.histogram('l6', lambdas_i6)
    tf.summary.histogram('l7', lambdas_i7)

    # 调整权重
    rate = 0.94
    lambdas_final = rate * lambdas_i + (1 - rate) / 6 * (lambdas_i2 + lambdas_i3 + lambdas_i4
                                                         + lambdas_i5 + lambdas_i6 + lambdas_i7)

    grads = tf.gradients(prediction, tf.trainable_variables(), grad_ys=tf.expand_dims(lambdas_final, 1))
    return ndcg, grads


def calc_multi_loss_gradient_v4(queries, labels, prediction, batch_size, features):
    with tf.variable_scope('lambda_loss'):
        q_ids, idx = tf.unique(tf.squeeze(queries))
        batch_labels = tf.dynamic_partition(labels, idx, batch_size)
        batch_predictions = tf.dynamic_partition(prediction, idx, batch_size)

        # lambda-loss部分
        losses_results = [lambda_loss2(_labels, _predictions)
                          for _labels, _predictions in zip(batch_labels, batch_predictions)]
        ndcg = tf.concat([tf.reshape(losses_results[i][0], [1]) for i in range(batch_size)], 0)
        ndcg = tf.gather_nd(ndcg, tf.where(tf.logical_not(tf.is_nan(ndcg))))
        # 保存的ndcg权重
        pair_weights = [losses_results[i][3] for i in range(batch_size)]

        # 这里加上利用距离-评分打标部分,暂时先不使用NDCG
        batch_scores = tf.dynamic_partition(features['avgScore'], idx, batch_size)
        batch_dis = tf.dynamic_partition(features['distance'], idx, batch_size)
        batch_show = tf.dynamic_partition(features['feaSubMetaCharLenBy1'], idx, batch_size)
        batch_rel = tf.dynamic_partition(features['querypoiNameSim'], idx, batch_size)

        ## 距离上的
        losses_results2 = [lambda_loss_for_distance(_labels, _predictions, _weight)
                           for _labels, _predictions, _weight in zip(batch_dis, batch_predictions, pair_weights)]
        ## 评分上的
        losses_results3 = [lambda_loss_for_score(_labels, _predictions, _weight)
                           for _labels, _predictions, _weight in zip(batch_scores, batch_predictions, pair_weights)]

        ## 展示相关性的
        losses_results4 = [lambda_loss_for_show(_labels, _predictions, _weight)
                           for _labels, _predictions, _weight in zip(batch_show, batch_predictions, pair_weights)]

        # poiname相关性
        losses_results5 = [lambda_loss_for_score(_labels, _predictions, _weight)
                           for _labels, _predictions, _weight in zip(batch_rel, batch_predictions, pair_weights)]

    lambdas_i = tf.concat([losses_results[i][2] for i in range(batch_size)], axis=0)
    lambdas_i = tf.where(tf.logical_not(tf.is_nan(lambdas_i)), lambdas_i, tf.zeros_like(lambdas_i))
    lambdas_i = lambdas_i / tf.maximum(1.0, tf.reduce_sum(tf.abs(lambdas_i)))

    lambdas_i2 = tf.concat(losses_results2, axis=0)
    lambdas_i2 = tf.where(tf.logical_not(tf.is_nan(lambdas_i2)), lambdas_i2, tf.zeros_like(lambdas_i2))
    lambdas_i2 = lambdas_i2 / tf.maximum(1.0, tf.reduce_sum(tf.abs(lambdas_i2)))

    lambdas_i3 = tf.concat(losses_results3, axis=0)
    lambdas_i3 = tf.where(tf.logical_not(tf.is_nan(lambdas_i3)), lambdas_i3, tf.zeros_like(lambdas_i3))
    lambdas_i3 = lambdas_i3 / tf.maximum(1.0, tf.reduce_sum(tf.abs(lambdas_i3)))

    lambdas_i4 = tf.concat(losses_results4, axis=0)
    lambdas_i4 = tf.where(tf.logical_not(tf.is_nan(lambdas_i4)), lambdas_i4, tf.zeros_like(lambdas_i4))
    lambdas_i4 = lambdas_i4 / tf.maximum(1.0, tf.reduce_sum(tf.abs(lambdas_i4)))

    lambdas_i5 = tf.concat(losses_results5, axis=0)
    lambdas_i5 = tf.where(tf.logical_not(tf.is_nan(lambdas_i5)), lambdas_i5, tf.zeros_like(lambdas_i5))
    lambdas_i5 = lambdas_i4 / tf.maximum(1.0, tf.reduce_sum(tf.abs(lambdas_i5)))

    tf.summary.histogram('l1', lambdas_i)
    tf.summary.histogram('l2', lambdas_i2)
    tf.summary.histogram('l3', lambdas_i3)
    tf.summary.histogram('l4', lambdas_i4)
    tf.summary.histogram('l5', lambdas_i5)

    # 调整权重
    rate = 0.98
    lambdas_final = rate * lambdas_i + (1 - rate) / 4 * (lambdas_i2 + lambdas_i3 + lambdas_i4 + lambdas_i5)

    grads = tf.gradients(prediction, tf.trainable_variables(), grad_ys=tf.expand_dims(lambdas_final, 1))
    return ndcg, grads


def tmp_post_lambdas(loss_result):
    lambdas_i3 = tf.concat(loss_result, axis=0)
    lambdas_i3 = tf.where(tf.logical_not(tf.is_nan(lambdas_i3)), lambdas_i3, tf.zeros_like(lambdas_i3))
    lambdas_i3 = lambdas_i3 / tf.maximum(1.0, tf.reduce_sum(tf.abs(lambdas_i3)))
    return lambdas_i3


def calc_lambda_multi_loss_gradient(queries, labels, predictions_tuple, batch_size):
    with tf.variable_scope('lambda_loss'):
        q_ids, idx = tf.unique(tf.squeeze(queries))
        batch_labels = tf.dynamic_partition(labels, idx, batch_size)

        ndcg_list = []
        grad_ys = 0.
        # 要手动调权重
        weights = [0.7, 0.1]

        for ind, prediction in enumerate(predictions_tuple):
            batch_predictions = tf.dynamic_partition(prediction, idx, batch_size)

            losses_results = [lambda_loss2(_labels, _predictions)
                              for _labels, _predictions in zip(batch_labels, batch_predictions)]
            ndcg = tf.concat([tf.reshape(losses_results[i][0], [1]) for i in range(batch_size)], 0)
            ndcg = tf.gather_nd(ndcg, tf.where(tf.logical_not(tf.is_nan(ndcg))))

            lambdas_i = tf.concat([losses_results[i][2] for i in range(batch_size)], axis=0)
            lambdas_i = tf.where(tf.logical_not(tf.is_nan(lambdas_i)), lambdas_i, tf.zeros_like(lambdas_i))
            lambdas_i = lambdas_i / tf.maximum(1.0, tf.reduce_sum(tf.abs(lambdas_i)))

            grad_ys = grad_ys + weights[ind] * tf.expand_dims(lambdas_i, 1)
            ndcg_list.append(ndcg)

    final_predictions = weights[0] * predictions_tuple[0] + weights[1] * predictions_tuple[1]
    grads = tf.gradients(final_predictions, tf.trainable_variables(), grad_ys=grad_ys)

    # 拿主分支ndcg
    ndcg = ndcg_list[0]

    return ndcg, grads


def lambda_loss(labels, predictions):
    with tf.variable_scope('lambda_loss'):
        lambdas, loss, s_ij_hat = calc_lambdas(labels, predictions)
        ndcg, delta_ndcg = calc_ndcg(labels, predictions)
        lambdas_ndcg = lambdas * delta_ndcg

        ij_positive_label_mat = tf.maximum(s_ij_hat, 0)
        ij_positive_mat_lambdas = ij_positive_label_mat * lambdas_ndcg
        ij_sum_lambdas = tf.reduce_sum(ij_positive_mat_lambdas, [1])
        ji_sum_lambdas = tf.reduce_sum(ij_positive_mat_lambdas, [0])
        lambdas_i = ij_sum_lambdas - ji_sum_lambdas

        return ndcg, loss, lambdas_i


def calc_lambdas(labels, predictions):
    with tf.variable_scope('lambdas'):
        sigma = 1
        s_ij = predictions - tf.transpose(predictions)

        # 主要是这里的s_ij_hat做修复
        tmp = labels - tf.transpose(labels)
        s_ij_hat = tf.where(tf.greater(tmp, 0),
                            tf.ones_like(tmp),
                            tf.where(tf.less(tmp, 0), tf.ones_like(tmp) * (-1.0), tf.zeros_like(tmp)))

        sigma_ij = tf.divide(1.0, tf.exp(tf.log(1 + tf.exp(-tf.abs(-sigma * s_ij))) -
                                         tf.minimum(0.0, -sigma * s_ij)))
        lambdas = sigma * (1.0 / 2.0) * (1 - s_ij_hat) - sigma * sigma_ij

        p_ij_hat = 1.0 / 2.0 * (1 + s_ij_hat)
        c_ij = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=s_ij,
                                                                      labels=p_ij_hat))
        return lambdas, c_ij, s_ij_hat


def calc_ndcg(labels, predictions):
    with tf.variable_scope('ndcg'):
        n_data = tf.size(labels)
        labels = tf.squeeze(labels, 1)
        predictions = tf.squeeze(predictions, 1)
        # labels = tf.Print(labels, [labels], "labels::", summarize=1000)
        # predictions = tf.Print(predictions, [predictions], "predictions::", summarize=1000)
        sorted_relevance_labels, _ = tf.nn.top_k(labels, k=n_data, sorted=True)
        _, indices = tf.nn.top_k(predictions, k=n_data, sorted=True)
        relevance_labels = tf.reshape(tf.gather(labels, indices), [n_data, 1])
        sorted_relevance_labels = tf.reshape(sorted_relevance_labels, [n_data, 1])
        index_range = tf.reshape(tf.range(tf.cast(n_data, tf.float32)), [n_data, 1])

        cg_discount = tf.log(index_range + 2) / tf.log(tf.constant(2.0, dtype=tf.float32))
        dcg_each = (2 ** relevance_labels - 1) / cg_discount
        dcg = tf.reduce_sum(dcg_each)
        idcg_each = (2 ** sorted_relevance_labels - 1) / cg_discount
        idcg = tf.reduce_sum(idcg_each)
        ndcg = dcg / idcg

        stale_ij = tf.tile(dcg_each, [1, n_data])
        new_ij = ((2 ** relevance_labels - 1) / tf.transpose(cg_discount))
        stale_ji = tf.tile(tf.transpose(dcg_each), [n_data, 1])
        new_ji = (tf.transpose(2 ** relevance_labels - 1) / cg_discount)
        new_ndcg = (dcg - stale_ij + new_ij - stale_ji + new_ji) / idcg
        delta_ndcg = tf.abs(ndcg - new_ndcg)
        return ndcg, delta_ndcg


def lambda_loss2(labels, predictions):
    with tf.variable_scope('lambda_loss'):
        lambdas, loss, s_ij_hat = calc_lambdas(labels, predictions)
        ndcg, pair_weight = calc_ndcg_loss2(labels, predictions)
        lambdas_ndcg = lambdas * pair_weight

        ij_positive_label_mat = tf.maximum(s_ij_hat, 0)
        ij_positive_mat_lambdas = ij_positive_label_mat * lambdas_ndcg
        ij_sum_lambdas = tf.reduce_sum(ij_positive_mat_lambdas, [1])
        ji_sum_lambdas = tf.reduce_sum(ij_positive_mat_lambdas, [0])
        lambdas_i = ij_sum_lambdas - ji_sum_lambdas

        return ndcg, loss, lambdas_i, pair_weight


# 只需要返回lambdas_i即可
def lambda_loss_for_distance(pre_labels, predictions, pair_weights):
    with tf.variable_scope('entropy_loss'):
        lambdas, loss, s_ij_hat = calc_lambdas_for_distance(pre_labels, predictions)

        ij_positive_label_mat = tf.maximum(s_ij_hat, 0)
        ij_positive_mat_lambdas = ij_positive_label_mat * (lambdas * pair_weights)
        ij_sum_lambdas = tf.reduce_sum(ij_positive_mat_lambdas, [1])
        ji_sum_lambdas = tf.reduce_sum(ij_positive_mat_lambdas, [0])
        lambdas_i = ij_sum_lambdas - ji_sum_lambdas

        return lambdas_i


def lambda_loss_for_score(pre_labels, predictions, pair_weights):
    with tf.variable_scope('entropy_loss'):
        lambdas, loss, s_ij_hat = calc_lambdas_for_score(pre_labels, predictions)

        ij_positive_label_mat = tf.maximum(s_ij_hat, 0)
        ij_positive_mat_lambdas = ij_positive_label_mat * (lambdas * pair_weights)
        ij_sum_lambdas = tf.reduce_sum(ij_positive_mat_lambdas, [1])
        ji_sum_lambdas = tf.reduce_sum(ij_positive_mat_lambdas, [0])
        lambdas_i = ij_sum_lambdas - ji_sum_lambdas

        return lambdas_i


def lambda_loss_for_show(pre_labels, predictions, pair_weights):
    with tf.variable_scope('entropy_loss'):
        lambdas, loss, s_ij_hat = calc_lambdas_for_show(pre_labels, predictions)

        ij_positive_label_mat = tf.maximum(s_ij_hat, 0)
        ij_positive_mat_lambdas = ij_positive_label_mat * (lambdas * pair_weights)
        ij_sum_lambdas = tf.reduce_sum(ij_positive_mat_lambdas, [1])
        ji_sum_lambdas = tf.reduce_sum(ij_positive_mat_lambdas, [0])
        lambdas_i = ij_sum_lambdas - ji_sum_lambdas

        return lambdas_i


# 针对sparse特征的lambda函数
# target参数是正样本的标签
def lambda_loss_for_sparse(pre_labels, predictions, pair_weights, target):
    with tf.variable_scope('entropy_loss'):
        lambdas, loss, s_ij_hat = calc_lambdas_for_sparse(pre_labels, predictions, target)

        ij_positive_label_mat = tf.maximum(s_ij_hat, 0)
        ij_positive_mat_lambdas = ij_positive_label_mat * (lambdas * pair_weights)
        ij_sum_lambdas = tf.reduce_sum(ij_positive_mat_lambdas, [1])
        ji_sum_lambdas = tf.reduce_sum(ij_positive_mat_lambdas, [0])
        lambdas_i = ij_sum_lambdas - ji_sum_lambdas

        return lambdas_i


# 注意距离和评分构造label时是相反的
def calc_lambdas_for_distance(pre_labels, predictions):
    with tf.variable_scope('lambdas'):
        sigma = 1
        s_ij = predictions - tf.transpose(predictions)
        tmp = pre_labels - tf.transpose(pre_labels)
        s_ij_hat = tf.where(tf.less(tmp, 0), tf.ones_like(tmp),
                            tf.where(tf.greater(tmp, 0), tf.ones_like(tmp) * (-1), tf.zeros_like(tmp)))
        s_ij_hat = tf.cast(s_ij_hat, tf.float32)
        sigma_ij = tf.divide(1.0, tf.exp(tf.log(1 + tf.exp(-tf.abs(-sigma * s_ij))) -
                                         tf.minimum(0.0, -sigma * s_ij)))
        lambdas = sigma * (1.0 / 2.0) * (1 - s_ij_hat) - sigma * sigma_ij

        p_ij_hat = 1.0 / 2.0 * (1 + s_ij_hat)
        c_ij = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=s_ij,
                                                                      labels=p_ij_hat))
        return lambdas, c_ij, s_ij_hat


def calc_lambdas_for_score(pre_labels, predictions):
    with tf.variable_scope('lambdas'):
        sigma = 1
        s_ij = predictions - tf.transpose(predictions)
        tmp = pre_labels - tf.transpose(pre_labels)
        s_ij_hat = tf.where(tf.greater(tmp, 0), tf.ones_like(tmp),
                            tf.where(tf.less(tmp, 0), tf.ones_like(tmp) * (-1), tf.zeros_like(tmp)))
        s_ij_hat = tf.cast(s_ij_hat, tf.float32)
        sigma_ij = tf.divide(1.0, tf.exp(tf.log(1 + tf.exp(-tf.abs(-sigma * s_ij))) -
                                         tf.minimum(0.0, -sigma * s_ij)))
        lambdas = sigma * (1.0 / 2.0) * (1 - s_ij_hat) - sigma * sigma_ij

        p_ij_hat = 1.0 / 2.0 * (1 + s_ij_hat)
        c_ij = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=s_ij,
                                                                      labels=p_ij_hat))
        return lambdas, c_ij, s_ij_hat


def calc_lambdas_for_show(pre_labels, predictions):
    with tf.variable_scope('lambdas'):
        sigma = 1
        s_ij = predictions - tf.transpose(predictions)
        tmp = pre_labels - tf.transpose(pre_labels)
        s_ij_hat = tf.where(tf.greater(tmp, 0), tf.ones_like(tmp),
                            tf.where(tf.less(tmp, 0), tf.ones_like(tmp) * (-1), tf.zeros_like(tmp)))
        s_ij_hat = tf.cast(s_ij_hat, tf.float32)
        sigma_ij = tf.divide(1.0, tf.exp(tf.log(1 + tf.exp(-tf.abs(-sigma * s_ij))) -
                                         tf.minimum(0.0, -sigma * s_ij)))
        lambdas = sigma * (1.0 / 2.0) * (1 - s_ij_hat) - sigma * sigma_ij

        p_ij_hat = 1.0 / 2.0 * (1 + s_ij_hat)
        c_ij = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=s_ij,
                                                                      labels=p_ij_hat))
        return lambdas, c_ij, s_ij_hat


def calc_lambdas_for_sparse(pre_labels, predictions, target):
    with tf.variable_scope('lambdas'):
        sigma = 1
        s_ij = predictions - tf.transpose(predictions)

        # 利用target构造,非1即0,且1表示正样本
        tmp = tf.where(tf.equal(pre_labels, target), tf.ones_like(pre_labels),
                       tf.zeros_like(pre_labels))

        # 类型转换
        tmp = tf.cast(tmp, tf.float32)

        s_ij_hat = tf.minimum(1.0, tf.maximum(-1.0, tmp - tf.transpose(tmp)))

        s_ij_hat = tf.cast(s_ij_hat, tf.float32)
        sigma_ij = tf.divide(1.0, tf.exp(tf.log(1 + tf.exp(-tf.abs(-sigma * s_ij))) -
                                         tf.minimum(0.0, -sigma * s_ij)))
        lambdas = sigma * (1.0 / 2.0) * (1 - s_ij_hat) - sigma * sigma_ij

        p_ij_hat = 1.0 / 2.0 * (1 + s_ij_hat)
        c_ij = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=s_ij,
                                                                      labels=p_ij_hat))
        return lambdas, c_ij, s_ij_hat


def calc_ndcg_loss2(labels, predictions):
    # lambdas, loss, s_ij_hat = calc_lambdas(labels, predictions)
    ndcg, delta_ndcg = calc_delta_ndcg(labels, predictions)
    n_data = tf.size(labels)

    idcg = max_dcg(labels, n_data)
    gain = (2 ** labels - 1) / idcg

    gain = tf.reshape(gain, [1, -1])
    pair_gain = array_ops.expand_dims(gain, 2) - array_ops.expand_dims(gain, 1)
    pair_weight = weight_for_relative_rank_diff(labels, predictions, pair_gain)
    pair_weight = tf.squeeze(pair_weight, axis=0)
    return ndcg, pair_weight


def calc_delta_ndcg(labels, predictions):
    with tf.variable_scope('ndcg'):
        n_data = tf.size(labels)
        return ndcg_k(labels, predictions, n_data)


def ndcg_k(labels, predictions, k):
    labels = tf.reshape(labels, [-1])
    doc_size = tf.size(predictions)
    k = tf.cond(tf.greater(k, doc_size),
                lambda: doc_size,
                lambda: k)
    predictions = tf.reshape(predictions, [-1])
    sorted_relevance_labels, _ = tf.nn.top_k(labels, k=k, sorted=True)
    _, indices = tf.nn.top_k(predictions, k=k, sorted=True)
    relevance_labels = tf.reshape(tf.gather(labels, indices), [-1, 1])
    sorted_relevance_labels = tf.reshape(sorted_relevance_labels, [-1, 1])

    index_range = tf.reshape(tf.range(tf.cast(k, tf.float32)), [-1, 1])
    cg_discount = tf.log(index_range + 2) / tf.log(tf.constant(2.0, dtype=tf.float32))
    dcg_each = (2 ** relevance_labels - 1) / cg_discount
    dcg = tf.reduce_sum(dcg_each)
    idcg_each = (2 ** sorted_relevance_labels - 1) / cg_discount
    idcg = tf.reduce_sum(idcg_each)
    ndcg = dcg / idcg

    stale_ij = tf.tile(dcg_each, [1, k])
    new_ij = ((2 ** relevance_labels - 1) / tf.transpose(cg_discount))
    stale_ji = tf.tile(tf.transpose(dcg_each), [k, 1])
    new_ji = (tf.transpose(2 ** relevance_labels - 1) / cg_discount)
    new_ndcg = (dcg - stale_ij + new_ij - stale_ji + new_ji) / idcg
    delta_ndcg = tf.abs(ndcg - new_ndcg)

    return ndcg, delta_ndcg


def weight_for_relative_rank_diff(_labels, _predictions, pair_gain):
    """Rank-based discount in the LambdaLoss paper."""
    # The LambdaLoss is not well defined when topn is active and topn <
    # list_size. We cap the rank of examples to topn + 1 so that the rank
    # differene is capped to topn. This is just a convenient upperbound
    # when topn is active. We need to revisit this.
    list_size = array_ops.shape(_labels)[0]

    # topn = 10 or list_size
    topn = list_size
    rank = math_ops.range(list_size) + 1

    capped_rank = array_ops.where(
        math_ops.greater(rank, topn),
        array_ops.ones_like(rank) * (topn + 1), rank)
    rank_diff = math_ops.to_float(
        math_ops.abs(
            array_ops.expand_dims(capped_rank, 1) -
            array_ops.expand_dims(capped_rank, 0)))
    rank_diff_discount = tf.log(rank_diff + 1) / tf.log(tf.constant(2.0, dtype=tf.float32))
    rank_diff_discount_1 = tf.log(rank_diff + 2) / tf.log(tf.constant(2.0, dtype=tf.float32))
    pair_discount = array_ops.where(
        math_ops.greater(rank_diff, 0),
        math_ops.abs(
            1 / rank_diff_discount -
            1 / (rank_diff_discount_1 + 1)),
        array_ops.zeros_like(rank_diff))
    pair_weight = math_ops.abs(pair_gain) * pair_discount
    if topn is None:
        return pair_weight
    pair_mask = math_ops.logical_or(
        array_ops.expand_dims(math_ops.less_equal(rank, topn), 1),
        array_ops.expand_dims(math_ops.less_equal(rank, topn), 0))
    return pair_weight * math_ops.to_float(pair_mask)


def max_dcg(labels, k):
    """Computes the inverse of max DCG."""
    labels = tf.reshape(labels, [1, -1])
    ideal_sorted_labels, _ = tf.nn.top_k(labels, k=k, sorted=True)
    index_range = tf.reshape(tf.range(tf.cast(k, tf.float32)), [-1, 1])
    cg_discount = tf.log(index_range + 2) / tf.log(tf.constant(2.0, dtype=tf.float32))
    idcg_each = (2 ** ideal_sorted_labels - 1) / cg_discount
    idcg = tf.reduce_sum(idcg_each)
    return idcg


