#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from enum import Enum, unique
from typing import Dict, List

from torchrec.distributed.embedding_types import ShardedEmbeddingTable
from torchrec.modules.embedding_configs import DATA_TYPE_NUM_BITS, DataType


@unique
class EmbDimBucketerPolicy(Enum):
    """
    Config to specify how to bucketize embedding tables based on dimensions.

    single_bucket: All embedding tables are put into a single bucket.
    all_buckets: All the embedding tables with the same dim size are put in the same bucket.
    cacheline_buckets: All the embedding tables with the same dim cacheline size are put in the same bucket.
    """

    SINGLE_BUCKET = "single_bucket"
    ALL_BUCKETS = "all_buckets"
    CACHELINE_BUCKETS = "cacheline_buckets"


class EmbDimBucketer:
    """
    Buckets embedding dimensions into different groups based on their sizes. This is intended to be leveraged
    once planning is done, and at the sharding stage, per rank.

    The rationale to use bucketization is

     - When UVM_CACHING is used: FBGEMM Table Batched Embedding Operator supports a software managed cache for the embeddings placed on UVM (Host memory).
       However, the cache uses maximum embedding dim of all the tables batched in the operator as its unit of allocation. This results in waisted HBM memory,
       and higher miss rate, hence lower performance. Bucketizing can address this issue, allowing for higher effective cache size and better performace.

    - When all tables are placed on HBM: When tables with widely different embedding dimension are batched together, the register allocation in GPU will
      be mainly decided by the table with largest embedding dimension. This can lead to worse performance due to lower number of threads and lower occupancy.

    Note that Column wise sharding also to some extent addresses this problem, but has its own limitations.


    Generally, we expect the CACHELINE_BUCKETS policy perform better than ALL_BUCKETS, as it addresses the main issues and limits the number of buckets.


    Args:
        embedding_tables (List[ShardedEmbeddingTable]): list of sharded embedding
        cfg (EmbDimBucketerPolicy): Bucketing policy

    returns:
        emb_dim_buckets (Dict[int, int]): Mapping from embedding dim to bucket id


    Example:
        emb_dim_bucketer = EmbDimBucketer(embedding_tables, EmbDimBucketerPolicy.SINGLE_BUCKET)
        ...
        bucket = emb_dim_bucketer.get_bucket(embedding_tables[0], embedding_tables[0].data_type) # bucket table 0 is assigned to.
    """

    def __init__(
        self, embedding_tables: List[ShardedEmbeddingTable], cfg: EmbDimBucketerPolicy
    ) -> None:
        self.embedding_dim_buckets: Dict[int, int] = {}
        self.num_buckets = 1
        self.cacheline = 128
        if cfg == EmbDimBucketerPolicy.CACHELINE_BUCKETS:
            self.emb_dim_buckets: Dict[int, int] = self.cacheline_emb_buckets(
                embedding_tables
            )
        elif cfg == EmbDimBucketerPolicy.ALL_BUCKETS:
            self.emb_dim_buckets: Dict[int, int] = self.all_emb_buckets(
                embedding_tables
            )
        elif cfg == EmbDimBucketerPolicy.SINGLE_BUCKET:
            self.emb_dim_buckets: Dict[int, int] = self.single_emb_bucket(
                embedding_tables
            )
        else:
            AssertionError(f"Invalid bucketization config {cfg}")

    def bucket_count(self) -> int:
        return self.num_buckets

    def get_bucket(self, embedding_dim: int, dtype: DataType) -> int:
        if self.num_buckets == 1:
            return 0
        else:
            return self.bucket(embedding_dim, dtype)

    def single_emb_bucket(
        self,
        embedding_tables: List[ShardedEmbeddingTable],
    ) -> Dict[int, int]:
        buckets: Dict[int, int] = {}
        bucket_id = 0

        for table in embedding_tables:
            dim_in_bytes = self.dim_in_bytes(table.local_cols, table.data_type)
            buckets[dim_in_bytes] = bucket_id

        self.num_buckets = 1

        return buckets

    def all_emb_buckets(
        self,
        embedding_tables: List[ShardedEmbeddingTable],
    ) -> Dict[int, int]:
        buckets: Dict[int, int] = {}
        bucket_id = -1

        for table in embedding_tables:
            dim_in_bytes = self.dim_in_bytes(table.local_cols, table.data_type)
            if dim_in_bytes not in buckets.keys():
                bucket_id += 1
                buckets[dim_in_bytes] = bucket_id

        self.num_buckets = bucket_id + 1  # id starts from 0

        return buckets

    def cacheline_emb_buckets(
        self,
        embedding_tables: List[ShardedEmbeddingTable],
    ) -> Dict[int, int]:
        buckets: Dict[int, int] = {}
        cl_buckets: Dict[int, int] = {}
        bucket_id = -1

        for table in embedding_tables:
            dim_in_bytes = self.dim_in_bytes(table.local_cols, table.data_type)
            cl_dim = dim_in_bytes // self.cacheline
            if cl_dim not in cl_buckets.keys():
                bucket_id += 1
                cl_buckets[cl_dim] = bucket_id

            if dim_in_bytes not in buckets.keys():
                buckets[dim_in_bytes] = cl_buckets[cl_dim]

        self.num_buckets = bucket_id + 1  # id starts from 0

        return buckets

    def bucket(self, dim: int, dtype: DataType) -> int:
        return self.emb_dim_buckets[self.dim_in_bytes(dim, dtype)]

    def dim_in_bytes(self, dim: int, dtype: DataType) -> int:
        return dim * DATA_TYPE_NUM_BITS[dtype] // 8
