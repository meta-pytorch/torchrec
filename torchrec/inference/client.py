#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging

import grpc
import predictor_pb2
import predictor_pb2_grpc
import torch
from torchrec.datasets.criteo import DEFAULT_CAT_NAMES, DEFAULT_INT_NAMES
from torchrec.datasets.random import RandomRecDataset
from torchrec.datasets.utils import Batch


def create_training_batch(args: argparse.Namespace) -> Batch:
    return next(
        iter(
            RandomRecDataset(
                keys=DEFAULT_CAT_NAMES,
                batch_size=args.batch_size,
                hash_size=args.num_embedding_features,
                ids_per_feature=1,
                num_dense=len(DEFAULT_INT_NAMES),
            ),
        )
    )


def create_request(
    batch: Batch, args: argparse.Namespace
) -> predictor_pb2.PredictionRequest:
    def to_bytes(tensor: torch.Tensor) -> bytes:
        return tensor.cpu().numpy().tobytes()

    float_features = predictor_pb2.FloatFeatures(
        num_features=args.num_float_features,
        values=to_bytes(batch.dense_features),
    )

    id_list_features = predictor_pb2.SparseFeatures(
        num_features=args.num_id_list_features,
        values=to_bytes(batch.sparse_features.values()),
        lengths=to_bytes(batch.sparse_features.lengths()),
    )

    id_score_list_features = predictor_pb2.SparseFeatures(num_features=0)
    embedding_features = predictor_pb2.FloatFeatures(num_features=0)
    unary_features = predictor_pb2.SparseFeatures(num_features=0)

    return predictor_pb2.PredictionRequest(
        batch_size=args.batch_size,
        float_features=float_features,
        id_list_features=id_list_features,
        id_score_list_features=id_score_list_features,
        embedding_features=embedding_features,
        unary_features=unary_features,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ip",
        type=str,
        default="0.0.0.0",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=50051,
    )
    parser.add_argument(
        "--num_float_features",
        type=int,
        default=13,
    )
    parser.add_argument(
        "--num_id_list_features",
        type=int,
        default=26,
    )
    parser.add_argument(
        "--num_id_score_list_features",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--num_embedding_features",
        type=int,
        default=100000,
    )
    parser.add_argument(
        "--embedding_feature_dim",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
    )

    args: argparse.Namespace = parser.parse_args()

    training_batch: Batch = create_training_batch(args)
    request: predictor_pb2.PredictionRequest = create_request(training_batch, args)

    with grpc.insecure_channel(f"{args.ip}:{args.port}") as channel:
        stub = predictor_pb2_grpc.PredictorStub(channel)
        response = stub.Predict(request)
        print("Response: ", response.predictions["default"].data)

if __name__ == "__main__":
    logging.basicConfig()
