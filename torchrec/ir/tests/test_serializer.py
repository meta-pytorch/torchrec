#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

#!/usr/bin/env python3

import copy
import unittest
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch import nn
from torch.fx.passes.utils.fuser_utils import fuse_by_partitions
from torchrec.ir.serializer import JsonSerializer
from torchrec.ir.utils import (
    decapsulate_ir_modules,
    encapsulate_ir_modules,
    mark_dynamic_kjt,
    qualname,
)
from torchrec.modules.embedding_configs import data_type_to_dtype, EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.modules.feature_processor_ import (
    PositionWeightedModule,
    PositionWeightedModuleCollection,
)
from torchrec.modules.fp_embedding_modules import FeatureProcessedEmbeddingBagCollection
from torchrec.modules.regroup import KTRegroupAsDict
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor
from torchrec.types import DataType


class CompoundModule(nn.Module):
    def __init__(
        self,
        ebc: EmbeddingBagCollection,
        comp: Optional["CompoundModule"] = None,
        mlist: List[Union[EmbeddingBagCollection, "CompoundModule"]] = [],
    ) -> None:
        super().__init__()
        self.ebc = ebc
        self.comp = comp
        self.list = nn.ModuleList(mlist)

    def forward(self, features: KeyedJaggedTensor) -> List[torch.Tensor]:
        res = self.comp(features) if self.comp else []
        res.append(self.ebc(features).values())
        for m in self.list:
            if isinstance(m, CompoundModule):
                res.extend(m(features))
            else:
                res.append(m(features).values())
        return res


class CompoundModuleSerializer(JsonSerializer):
    _module_cls = CompoundModule

    @classmethod
    def children(cls, module: nn.Module) -> List[str]:
        children = ["ebc", "list"]
        if module.comp is not None:
            children += ["comp"]
        return children

    @classmethod
    def serialize_to_dict(
        cls,
        module: nn.Module,
    ) -> Dict[str, Any]:
        return {}

    @classmethod
    def deserialize_from_dict(
        cls,
        metadata_dict: Dict[str, Any],
        device: Optional[torch.device] = None,
        unflatten_ep: Optional[nn.Module] = None,
    ) -> nn.Module:
        assert unflatten_ep is not None
        ebc = unflatten_ep.ebc
        comp = getattr(unflatten_ep, "comp", None)
        i = 0
        mlist = []
        while hasattr(unflatten_ep.list, str(i)):
            mlist.append(getattr(unflatten_ep.list, str(i)))
            i += 1
        #  `Union[Module, Tensor]`.
        #  got `Union[Module, Tensor]`.
        return CompoundModule(ebc, comp, mlist)


class TestJsonSerializer(unittest.TestCase):
    # in the model we have 5 duplicated EBCs, 1 fpEBC with fpCollection, and 1 fpEBC with fpDict
    def generate_model(self) -> nn.Module:
        class Model(nn.Module):
            def __init__(self, ebc, fpebc1, fpebc2):
                super().__init__()
                self.ebc1 = ebc
                self.ebc2 = copy.deepcopy(ebc)
                self.ebc3 = copy.deepcopy(ebc)
                self.ebc4 = copy.deepcopy(ebc)
                self.ebc5 = copy.deepcopy(ebc)
                self.fpebc1 = fpebc1
                self.fpebc2 = fpebc2

            def forward(
                self,
                features: KeyedJaggedTensor,
            ) -> List[torch.Tensor]:
                kt1 = self.ebc1(features)
                kt2 = self.ebc2(features)
                kt3 = self.ebc3(features)
                kt4 = self.ebc4(features)
                kt5 = self.ebc5(features)

                fpebc1_res = self.fpebc1(features)
                fpebc2_res = self.fpebc2(features)
                res: List[torch.Tensor] = []
                for kt in [kt1, kt2, kt3, kt4, kt5, fpebc1_res, fpebc2_res]:
                    res.extend(KeyedTensor.regroup([kt], [[key] for key in kt.keys()]))
                return res

        tb1_config = EmbeddingBagConfig(
            name="t1",
            embedding_dim=3,
            num_embeddings=10,
            feature_names=["f1"],
        )
        tb2_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=4,
            num_embeddings=10,
            feature_names=["f2"],
        )
        tb3_config = EmbeddingBagConfig(
            name="t3",
            embedding_dim=5,
            num_embeddings=10,
            feature_names=["f3"],
        )

        ebc = EmbeddingBagCollection(
            tables=[tb1_config, tb2_config, tb3_config],
            is_weighted=False,
        )
        max_feature_lengths = {"f1": 100, "f2": 100}

        fpebc1 = FeatureProcessedEmbeddingBagCollection(
            EmbeddingBagCollection(
                tables=[tb1_config, tb2_config],
                is_weighted=True,
            ),
            PositionWeightedModuleCollection(
                max_feature_lengths=max_feature_lengths,
            ),
        )
        fpebc2 = FeatureProcessedEmbeddingBagCollection(
            EmbeddingBagCollection(
                tables=[tb1_config, tb3_config],
                is_weighted=True,
            ),
            {
                "f1": PositionWeightedModule(max_feature_length=10),
                "f3": PositionWeightedModule(max_feature_length=20),
            },
        )

        model = Model(ebc, fpebc1, fpebc2)

        return model

    def generate_model_for_vbe_kjt(self) -> nn.Module:
        class Model(nn.Module):
            def __init__(self, ebc):
                super().__init__()
                self.ebc1 = ebc

            def forward(
                self,
                features: KeyedJaggedTensor,
            ) -> List[torch.Tensor]:
                kt1 = self.ebc1(features)
                res: List[torch.Tensor] = []

                for kt in [kt1]:
                    res.extend(KeyedTensor.regroup([kt], [[key] for key in kt.keys()]))

                return res

        config1 = EmbeddingBagConfig(
            name="t1",
            embedding_dim=3,
            num_embeddings=10,
            feature_names=["f1"],
        )
        config2 = EmbeddingBagConfig(
            name="t2",
            embedding_dim=4,
            num_embeddings=10,
            feature_names=["f2"],
        )
        config3 = EmbeddingBagConfig(
            name="t3",
            embedding_dim=5,
            num_embeddings=10,
            feature_names=["f3"],
        )
        ebc = EmbeddingBagCollection(
            tables=[config1, config2, config3],
            is_weighted=False,
        )

        model = Model(ebc)

        return model

    def test_serialize_deserialize_ebc(self) -> None:
        model = self.generate_model()
        id_list_features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f2", "f3"],
            values=torch.tensor([0, 1, 2, 3, 2, 3]),
            offsets=torch.tensor([0, 2, 2, 3, 4, 5, 6]),
        )

        eager_out = model(id_list_features)

        # Serialize EBC
        model, sparse_fqns = encapsulate_ir_modules(model, JsonSerializer)
        ep = torch.export.export(
            model,
            (id_list_features,),
            {},
            strict=False,
            # Allows KJT to not be unflattened and run a forward on unflattened EP
            preserve_module_call_signature=(tuple(sparse_fqns)),
        )

        # Run forward on ExportedProgram
        ep_output = ep.module()(id_list_features)

        for i, tensor in enumerate(ep_output):
            self.assertEqual(eager_out[i].shape, tensor.shape)

        # Deserialize EBC
        unflatten_ep = torch.export.unflatten(ep)
        deserialized_model = decapsulate_ir_modules(unflatten_ep, JsonSerializer)

        # check EBC config
        for i in range(5):
            ebc_name = f"ebc{i + 1}"
            self.assertIsInstance(
                getattr(deserialized_model, ebc_name), EmbeddingBagCollection
            )

            for deserialized, orginal in zip(
                getattr(deserialized_model, ebc_name).embedding_bag_configs(),
                getattr(model, ebc_name).embedding_bag_configs(),
            ):
                self.assertEqual(deserialized.name, orginal.name)
                self.assertEqual(deserialized.embedding_dim, orginal.embedding_dim)
                self.assertEqual(deserialized.num_embeddings, orginal.num_embeddings)
                self.assertEqual(deserialized.feature_names, orginal.feature_names)

        # check FPEBC config
        for i in range(2):
            fpebc_name = f"fpebc{i + 1}"
            assert isinstance(
                getattr(deserialized_model, fpebc_name),
                FeatureProcessedEmbeddingBagCollection,
            )

            for deserialized, orginal in zip(
                getattr(
                    deserialized_model, fpebc_name
                )._embedding_bag_collection.embedding_bag_configs(),
                getattr(
                    model, fpebc_name
                )._embedding_bag_collection.embedding_bag_configs(),
            ):
                self.assertEqual(deserialized.name, orginal.name)
                self.assertEqual(deserialized.embedding_dim, orginal.embedding_dim)
                self.assertEqual(deserialized.num_embeddings, orginal.num_embeddings)
                self.assertEqual(deserialized.feature_names, orginal.feature_names)

        # Run forward on deserialized model and compare the output
        deserialized_model.load_state_dict(model.state_dict())
        deserialized_out = deserialized_model(id_list_features)

        self.assertEqual(len(deserialized_out), len(eager_out))
        for deserialized, orginal in zip(deserialized_out, eager_out):
            self.assertEqual(deserialized.shape, orginal.shape)
            torch.testing.assert_close(deserialized, orginal)

    def test_serialize_deserialize_ebc_with_vbe_kjt(self) -> None:
        model = self.generate_model_for_vbe_kjt()
        kjt_1 = KeyedJaggedTensor(
            keys=["f1", "f2", "f3"],
            values=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 0]),
            lengths=torch.tensor([1, 2, 3, 2, 1, 1]),
            stride_per_key_per_rank=torch.tensor([[3], [2], [1]]),
            inverse_indices=(
                ["f1", "f2", "f3"],
                torch.tensor([[0, 1, 2], [0, 1, 0], [0, 0, 0]]),
            ),
        )
        kjt_2 = KeyedJaggedTensor(
            keys=["f1", "f2", "f3"],
            values=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 0]),
            lengths=torch.tensor([1, 2, 3, 2, 1, 1]),
            stride_per_key_per_rank=torch.tensor([[1], [2], [3]]),
            inverse_indices=(
                ["f1", "f2", "f3"],
                torch.tensor([[0, 0, 0], [0, 1, 0], [0, 1, 2]]),
            ),
        )

        eager_out = model(kjt_1)
        eager_out_2 = model(kjt_2)

        # Serialize EBC
        model, sparse_fqns = encapsulate_ir_modules(model, JsonSerializer)
        ep = torch.export.export(
            model,
            (kjt_1,),
            {},
            strict=False,
            # Allows KJT to not be unflattened and run a forward on unflattened EP
            preserve_module_call_signature=(tuple(sparse_fqns)),
        )

        # Run forward on ExportedProgram
        ep_output = ep.module()(kjt_1)
        ep_output_2 = ep.module()(kjt_2)

        self.assertEqual(len(ep_output), len(kjt_1.keys()))
        self.assertEqual(len(ep_output_2), len(kjt_2.keys()))
        for i, tensor in enumerate(ep_output):
            self.assertEqual(eager_out[i].shape[1], tensor.shape[1])
        for i, tensor in enumerate(ep_output_2):
            self.assertEqual(eager_out_2[i].shape[1], tensor.shape[1])

        # Deserialize EBC
        unflatten_ep = torch.export.unflatten(ep)
        deserialized_model = decapsulate_ir_modules(unflatten_ep, JsonSerializer)

        # check EBC config
        for i in range(1):
            ebc_name = f"ebc{i + 1}"
            self.assertIsInstance(
                getattr(deserialized_model, ebc_name), EmbeddingBagCollection
            )

            for deserialized, orginal in zip(
                getattr(deserialized_model, ebc_name).embedding_bag_configs(),
                getattr(model, ebc_name).embedding_bag_configs(),
            ):
                self.assertEqual(deserialized.name, orginal.name)
                self.assertEqual(deserialized.embedding_dim, orginal.embedding_dim)
                self.assertEqual(deserialized.num_embeddings, orginal.num_embeddings)
                self.assertEqual(deserialized.feature_names, orginal.feature_names)

        # Run forward on deserialized model and compare the output
        deserialized_model.load_state_dict(model.state_dict())
        deserialized_out = deserialized_model(kjt_1)

        self.assertEqual(len(deserialized_out), len(eager_out))
        for deserialized, orginal in zip(deserialized_out, eager_out):
            self.assertEqual(deserialized.shape, orginal.shape)
            torch.testing.assert_close(deserialized, orginal)

        deserialized_out_2 = deserialized_model(kjt_2)

        self.assertEqual(len(deserialized_out_2), len(eager_out_2))
        for deserialized, orginal in zip(deserialized_out_2, eager_out_2):
            self.assertEqual(deserialized.shape, orginal.shape)
            torch.testing.assert_close(deserialized, orginal)

    def test_dynamic_shape_ebc_disabled_in_oss_compatibility(self) -> None:
        model = self.generate_model()
        feature1 = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f2", "f3"],
            values=torch.tensor([0, 1, 2, 3, 2, 3]),
            offsets=torch.tensor([0, 2, 2, 3, 4, 5, 6]),
        )

        feature2 = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f2", "f3"],
            values=torch.tensor([0, 1, 2, 3, 2, 3, 4]),
            offsets=torch.tensor([0, 2, 2, 3, 4, 5, 7]),
        )
        eager_out = model(feature2)

        # Serialize EBC
        collection = mark_dynamic_kjt(feature1)
        model, sparse_fqns = encapsulate_ir_modules(model, JsonSerializer)
        ep = torch.export.export(
            model,
            (feature1,),
            {},
            dynamic_shapes=collection.dynamic_shapes(model, (feature1,)),
            strict=False,
            # Allows KJT to not be unflattened and run a forward on unflattened EP
            preserve_module_call_signature=tuple(sparse_fqns),
        )

        # Run forward on ExportedProgram
        ep_output = ep.module()(feature2)

        # other asserts
        for i, tensor in enumerate(ep_output):
            self.assertEqual(eager_out[i].shape, tensor.shape)

        # Deserialize EBC
        unflatten_ep = torch.export.unflatten(ep)
        deserialized_model = decapsulate_ir_modules(unflatten_ep, JsonSerializer)
        deserialized_model.load_state_dict(model.state_dict())

        # Run forward on deserialized model
        deserialized_out = deserialized_model(feature2)

        for i, tensor in enumerate(deserialized_out):
            self.assertEqual(eager_out[i].shape, tensor.shape)
            torch.testing.assert_close(eager_out[i], tensor)

    def test_variable_batch_size_ebc_disabled_in_oss_compatibility(self) -> None:
        model = self.generate_model()
        feature1 = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f2", "f3"],
            values=torch.tensor([0, 1, 2, 3, 2, 3]),
            offsets=torch.tensor([0, 2, 2, 3, 4, 5, 6]),  # batch size = 2
        )

        feature2 = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f2", "f3"],
            values=torch.tensor([0, 1, 2, 3, 2, 3, 4, 5, 6]),
            offsets=torch.tensor([0, 2, 2, 3, 4, 5, 7, 8, 8, 9]),  # batch size = 3
        )
        eager_out1 = model(feature1)
        eager_out2 = model(feature2)
        # feature1.lengths()
        # feature2.lengths()

        # Serialize EBC with sample input (feature1, batch size = 2)
        collection = mark_dynamic_kjt(feature1, variable_batch=True)
        model, sparse_fqns = encapsulate_ir_modules(model, JsonSerializer)
        ep = torch.export.export(
            model,
            (feature1,),
            {},
            dynamic_shapes=collection.dynamic_shapes(model, (feature1,)),
            strict=False,
            # Allows KJT to not be unflattened and run a forward on unflattened EP
            preserve_module_call_signature=tuple(sparse_fqns),
        )

        # Run forward on ExportedProgram
        ep_output1 = ep.module()(feature1)
        ep_output2 = ep.module()(feature2)

        # other asserts
        for eager_out, ep_out in [(eager_out1, ep_output1), (eager_out2, ep_output2)]:
            for a, b in zip(eager_out, ep_out):
                self.assertEqual(a.shape, b.shape)

        # Deserialize EBC
        unflatten_ep = torch.export.unflatten(ep)
        deserialized_model = decapsulate_ir_modules(unflatten_ep, JsonSerializer)
        deserialized_model.load_state_dict(model.state_dict())

        # Run forward on deserialized model
        deserialized_out1 = deserialized_model(feature1)
        deserialized_out2 = deserialized_model(feature2)

        for e, d in ([eager_out1, deserialized_out1], [eager_out2, deserialized_out2]):
            for a, b in zip(e, d):
                self.assertEqual(a.shape, b.shape)
                torch.testing.assert_close(a, b)

    def test_ir_emb_lookup_device(self) -> None:
        model = self.generate_model()
        model.fpebc1 = copy.deepcopy(model.ebc1)
        model.fpebc2 = copy.deepcopy(model.ebc1)
        feature1 = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f2", "f3"],
            values=torch.tensor([0, 1, 2, 3, 2, 3]),
            offsets=torch.tensor([0, 2, 2, 3, 4, 5, 6]),
        )

        model, sparse_fqns = encapsulate_ir_modules(model, JsonSerializer)
        for device in ["cpu", "cuda", "meta"]:
            if device == "cuda" and not torch.cuda.is_available():
                continue
            device = torch.device(device)
            outputs = model.to(device)(feature1.to(device))
            for output in outputs:
                self.assertEqual(output.device.type, device.type)

    def test_deserialized_device(self) -> None:
        model = self.generate_model()
        id_list_features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f2", "f3"],
            values=torch.tensor([0, 1, 2, 3, 2, 3]),
            offsets=torch.tensor([0, 2, 2, 3, 4, 5, 6]),
        )

        # Serialize EBC
        model, sparse_fqns = encapsulate_ir_modules(model, JsonSerializer)
        ep = torch.export.export(
            model,
            (id_list_features,),
            {},
            strict=False,
            # Allows KJT to not be unflattened and run a forward on unflattened EP
            preserve_module_call_signature=(tuple(sparse_fqns)),
        )

        # Deserialize EBC on different devices (<cpu>, <cuda>, <meta>)
        for device in ["cpu", "cuda", "meta"]:
            if device == "cuda" and not torch.cuda.is_available():
                continue
            device = torch.device(device)
            unflatten_ep = torch.export.unflatten(ep)
            deserialized_model = decapsulate_ir_modules(
                unflatten_ep, JsonSerializer, device
            )
            for name, m in deserialized_model.named_modules():
                if hasattr(m, "device"):
                    assert m.device.type == device.type, f"{name} should be on {device}"
            for name, param in deserialized_model.named_parameters():
                # TODO: we don't support FPEBC yet, so we skip the FPEBC params
                if "_feature_processors" in name:
                    continue
                assert param.device.type == device.type, f"{name} should be on {device}"

    def test_compound_module(self) -> None:
        tb1_config = EmbeddingBagConfig(
            name="t1",
            embedding_dim=4,
            num_embeddings=10,
            feature_names=["f1"],
        )
        tb2_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=4,
            num_embeddings=10,
            feature_names=["f2"],
        )
        tb3_config = EmbeddingBagConfig(
            name="t3",
            embedding_dim=4,
            num_embeddings=10,
            feature_names=["f3"],
        )
        ebc: Callable[[], EmbeddingBagCollection] = lambda: EmbeddingBagCollection(
            tables=[tb1_config, tb2_config, tb3_config],
            is_weighted=False,
        )

        class MyModel(nn.Module):
            def __init__(self, comp: CompoundModule) -> None:
                super().__init__()
                self.comp = comp

            def forward(self, features: KeyedJaggedTensor) -> List[torch.Tensor]:
                return self.comp(features)

        model = MyModel(
            CompoundModule(
                ebc=ebc(),
                comp=CompoundModule(ebc(), CompoundModule(ebc(), mlist=[ebc(), ebc()])),
                mlist=[ebc(), CompoundModule(ebc(), CompoundModule(ebc()))],
            )
        )
        id_list_features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f2", "f3"],
            values=torch.tensor([0, 1, 2, 3, 2, 3]),
            offsets=torch.tensor([0, 2, 2, 3, 4, 5, 6]),
        )

        eager_out = model(id_list_features)

        JsonSerializer.module_to_serializer_cls[qualname(CompoundModule)] = (
            CompoundModuleSerializer
        )
        # Serialize
        model, sparse_fqns = encapsulate_ir_modules(model, JsonSerializer)
        ep = torch.export.export(
            model,
            (id_list_features,),
            {},
            strict=False,
            # Allows KJT to not be unflattened and run a forward on unflattened EP
            preserve_module_call_signature=(tuple(sparse_fqns)),
        )

        ep_output = ep.module()(id_list_features)
        self.assertEqual(len(ep_output), len(eager_out))
        for x, y in zip(ep_output, eager_out):
            self.assertEqual(x.shape, y.shape)

        # Deserialize
        unflatten_ep = torch.export.unflatten(ep)
        deserialized_model = decapsulate_ir_modules(unflatten_ep, JsonSerializer)
        # Check if Compound Module is deserialized correctly
        self.assertIsInstance(deserialized_model.comp, CompoundModule)
        self.assertIsInstance(deserialized_model.comp.comp, CompoundModule)
        self.assertIsInstance(deserialized_model.comp.comp.comp, CompoundModule)
        self.assertIsInstance(deserialized_model.comp.list[1], CompoundModule)
        self.assertIsInstance(deserialized_model.comp.list[1].comp, CompoundModule)

        deserialized_model.load_state_dict(model.state_dict())
        # Run forward on deserialized model
        deserialized_out = deserialized_model(id_list_features)
        self.assertEqual(len(deserialized_out), len(eager_out))
        for x, y in zip(deserialized_out, eager_out):
            torch.testing.assert_close(x, y)

    def test_regroup_as_dict_module(self) -> None:
        class Model(nn.Module):
            def __init__(self, ebc, fpebc, regroup):
                super().__init__()
                self.ebc = ebc
                self.fpebc = fpebc
                self.regroup = regroup

            def forward(
                self,
                features: KeyedJaggedTensor,
            ) -> Dict[str, torch.Tensor]:
                kt1 = self.ebc(features)
                kt2 = self.fpebc(features)
                return self.regroup([kt1, kt2])

        tb1_config = EmbeddingBagConfig(
            name="t1",
            embedding_dim=3,
            num_embeddings=10,
            feature_names=["f1", "f2"],
        )
        tb2_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=4,
            num_embeddings=10,
            feature_names=["f3", "f4"],
        )
        tb3_config = EmbeddingBagConfig(
            name="t3",
            embedding_dim=5,
            num_embeddings=10,
            feature_names=["f5"],
        )

        ebc = EmbeddingBagCollection(
            tables=[tb1_config, tb3_config],
            is_weighted=False,
        )
        max_feature_lengths = {"f3": 100, "f4": 100}
        fpebc = FeatureProcessedEmbeddingBagCollection(
            EmbeddingBagCollection(
                tables=[tb2_config],
                is_weighted=True,
            ),
            PositionWeightedModuleCollection(
                max_feature_lengths=max_feature_lengths,
            ),
        )
        regroup = KTRegroupAsDict([["f1", "f3", "f5"], ["f2", "f4"]], ["odd", "even"])
        model = Model(ebc, fpebc, regroup)

        id_list_features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f2", "f3", "f4", "f5"],
            values=torch.tensor([0, 1, 2, 3, 2, 3, 4, 5, 6, 7, 8, 9, 1, 1, 2]),
            offsets=torch.tensor([0, 2, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15]),
        )
        self.assertFalse(model.regroup._is_inited)

        # Serialize EBC
        model, sparse_fqns = encapsulate_ir_modules(model, JsonSerializer)
        ep = torch.export.export(
            model,
            (id_list_features,),
            {},
            strict=False,
            # Allows KJT to not be unflattened and run a forward on unflattened EP
            preserve_module_call_signature=(tuple(sparse_fqns)),
        )

        #  `_is_inited`.
        self.assertFalse(model.regroup._is_inited)
        eager_out = model(id_list_features)
        #  `_is_inited`.
        self.assertFalse(model.regroup._is_inited)

        # Run forward on ExportedProgram
        ep_output = ep.module()(id_list_features)
        for key in eager_out.keys():
            self.assertEqual(ep_output[key].shape, eager_out[key].shape)
        # Deserialize EBC
        unflatten_ep = torch.export.unflatten(ep)
        deserialized_model = decapsulate_ir_modules(unflatten_ep, JsonSerializer)
        #  `_is_inited`.
        self.assertFalse(deserialized_model.regroup._is_inited)
        deserialized_out = deserialized_model(id_list_features)
        #  `_is_inited`.
        self.assertTrue(deserialized_model.regroup._is_inited)
        for key in eager_out.keys():
            self.assertEqual(deserialized_out[key].shape, eager_out[key].shape)

    def test_key_order_with_ebc_and_regroup(self) -> None:
        tb1_config = EmbeddingBagConfig(
            name="t1",
            embedding_dim=3,
            num_embeddings=10,
            feature_names=["f1"],
        )
        tb2_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=4,
            num_embeddings=10,
            feature_names=["f2"],
        )
        tb3_config = EmbeddingBagConfig(
            name="t3",
            embedding_dim=5,
            num_embeddings=10,
            feature_names=["f3"],
        )
        id_list_features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f2", "f3", "f4", "f5"],
            values=torch.tensor([0, 1, 2, 3, 2, 3, 4, 5, 6, 7, 8, 9, 1, 1, 2]),
            offsets=torch.tensor([0, 2, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15]),
        )
        ebc1 = EmbeddingBagCollection(
            tables=[tb1_config, tb2_config, tb3_config],
            is_weighted=False,
        )
        ebc2 = EmbeddingBagCollection(
            tables=[tb1_config, tb3_config, tb2_config],
            is_weighted=False,
        )
        ebc2.load_state_dict(ebc1.state_dict())
        regroup = KTRegroupAsDict([["f1", "f3"], ["f2"]], ["odd", "even"])

        class mySparse(nn.Module):
            def __init__(self, ebc, regroup):
                super().__init__()
                self.ebc = ebc
                self.regroup = regroup

            def forward(
                self,
                features: KeyedJaggedTensor,
            ) -> Dict[str, torch.Tensor]:
                return self.regroup(keyed_tensors=[self.ebc(features)])

        class myModel(nn.Module):
            def __init__(self, ebc, regroup):
                super().__init__()
                self.sparse = mySparse(ebc, regroup)

            def forward(
                self,
                features: KeyedJaggedTensor,
            ) -> Dict[str, torch.Tensor]:
                return self.sparse(features)

        model = myModel(ebc1, regroup)
        eager_out = model(id_list_features)

        model, sparse_fqns = encapsulate_ir_modules(model, JsonSerializer)
        ep = torch.export.export(
            model,
            (id_list_features,),
            {},
            strict=False,
            # Allows KJT to not be unflattened and run a forward on unflattened EP
            preserve_module_call_signature=(tuple(sparse_fqns)),
        )
        unflatten_ep = torch.export.unflatten(ep)
        deserialized_model = decapsulate_ir_modules(
            unflatten_ep,
            JsonSerializer,
            short_circuit_pytree_ebc_regroup=True,
            finalize_interpreter_modules=True,
        )

        #  we export the model with ebc1 and unflatten the model,
        #  and then swap with ebc2 (you can think this as the the sharding process
        #  resulting a shardedEBC), so that we can mimic the key-order change
        deserialized_model.sparse.ebc = ebc2

        deserialized_out = deserialized_model(id_list_features)
        for key in eager_out.keys():
            torch.testing.assert_close(deserialized_out[key], eager_out[key])

    def test_key_order_with_ebc_and_regroup_input_kwargs(self) -> None:
        tb1_config = EmbeddingBagConfig(
            name="t1",
            embedding_dim=3,
            num_embeddings=10,
            feature_names=["f1"],
        )
        tb2_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=4,
            num_embeddings=10,
            feature_names=["f2"],
        )
        tb3_config = EmbeddingBagConfig(
            name="t3",
            embedding_dim=5,
            num_embeddings=10,
            feature_names=["f3"],
        )
        id_list_features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f2", "f3", "f4", "f5"],
            values=torch.tensor([0, 1, 2, 3, 2, 3, 4, 5, 6, 7, 8, 9, 1, 1, 2]),
            offsets=torch.tensor([0, 2, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15]),
        )
        ebc1 = EmbeddingBagCollection(
            tables=[tb1_config, tb2_config, tb3_config],
            is_weighted=False,
        )
        ebc2 = EmbeddingBagCollection(
            tables=[tb1_config, tb3_config, tb2_config],
            is_weighted=False,
        )
        ebc2.load_state_dict(ebc1.state_dict())
        regroup = KTRegroupAsDict([["f1", "f3"], ["f2"]], ["odd", "even"])

        class mySparse(nn.Module):
            def __init__(self, ebc):
                super().__init__()
                self.ebc = ebc

            def forward(
                self,
                features: KeyedJaggedTensor,
            ) -> KeyedTensor:
                return self.ebc(features)

        class myModel(nn.Module):
            def __init__(self, ebc, regroup):
                super().__init__()
                self.regroup = regroup
                self.sparse = mySparse(ebc)

            def forward(
                self,
                features: KeyedJaggedTensor,
            ) -> Dict[str, torch.Tensor]:
                sparse_out = self.sparse(features)
                return self.regroup(keyed_tensors=[sparse_out])

        model = myModel(ebc1, regroup)
        eager_out = model(id_list_features)

        model, sparse_fqns = encapsulate_ir_modules(model, JsonSerializer)
        ep = torch.export.export(
            model,
            (id_list_features,),
            {},
            strict=False,
            # Allows KJT to not be unflattened and run a forward on unflattened EP
            preserve_module_call_signature=(tuple(sparse_fqns)),
        )
        unflatten_ep = torch.export.unflatten(ep)
        deserialized_model = decapsulate_ir_modules(
            unflatten_ep,
            JsonSerializer,
            short_circuit_pytree_ebc_regroup=True,
            finalize_interpreter_modules=True,
        )

        #  we export the model with ebc1 and unflatten the model,
        #  and then swap with ebc2 (you can think this as the the sharding process
        #  resulting a shardedEBC), so that we can mimic the key-order change
        deserialized_model.sparse.ebc = ebc2

        deserialized_out = deserialized_model(id_list_features)
        for key in eager_out.keys():
            torch.testing.assert_close(deserialized_out[key], eager_out[key])

    def test_key_order_with_ebc_and_regroup_in_subgraph(self) -> None:
        tb1_config = EmbeddingBagConfig(
            name="t1",
            embedding_dim=3,
            num_embeddings=10,
            feature_names=["f1"],
        )
        tb2_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=4,
            num_embeddings=10,
            feature_names=["f2"],
        )
        tb3_config = EmbeddingBagConfig(
            name="t3",
            embedding_dim=5,
            num_embeddings=10,
            feature_names=["f3"],
        )
        id_list_features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f2", "f3", "f4", "f5"],
            values=torch.tensor([0, 1, 2, 3, 2, 3, 4, 5, 6, 7, 8, 9, 1, 1, 2]),
            offsets=torch.tensor([0, 2, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15]),
        )
        ebc1 = EmbeddingBagCollection(
            tables=[tb1_config, tb2_config, tb3_config],
            is_weighted=False,
        )
        ebc2 = EmbeddingBagCollection(
            tables=[tb1_config, tb3_config, tb2_config],
            is_weighted=False,
        )
        ebc2.load_state_dict(ebc1.state_dict())
        regroup = KTRegroupAsDict([["f1", "f3"], ["f2"]], ["odd", "even"])

        class mySparse(nn.Module):
            def __init__(self, ebc, regroup):
                super().__init__()
                self.ebc = ebc
                self.regroup = regroup

            def forward(
                self,
                features: KeyedJaggedTensor,
            ) -> Dict[str, torch.Tensor]:
                return self.regroup(keyed_tensors=[self.ebc(features)])

        class myModel(nn.Module):
            def __init__(self, ebc, regroup):
                super().__init__()
                self.sparse = mySparse(ebc, regroup)

            def forward(
                self,
                features: KeyedJaggedTensor,
            ) -> Dict[str, torch.Tensor]:
                return self.sparse(features)

        model = myModel(ebc1, regroup)
        eager_out = model(id_list_features)

        model, sparse_fqns = encapsulate_ir_modules(model, JsonSerializer)
        ep = torch.export.export(
            model,
            (id_list_features,),
            {},
            strict=False,
            # Allows KJT to not be unflattened and run a forward on unflattened EP
            preserve_module_call_signature=(tuple(sparse_fqns)),
        )
        unflatten_ep = torch.export.unflatten(ep)
        # Create subgraph for regroup node
        regroup_nodes = []
        mod = unflatten_ep.sparse
        for node in mod.graph.nodes:
            if node.op == "call_module" and "regroup" in str(node.target):
                regroup_nodes.append(node)
        partitions: List[Dict[torch.fx.Node, Optional[int]]] = []
        partitions.append(dict.fromkeys(regroup_nodes, None))
        fuse_by_partitions(unflatten_ep.sparse.graph_module, partitions)
        sub_mod = torch.fx.graph_module._get_attr(mod.graph_module, "fused_0")
        mod.add_module("fused_0", sub_mod)
        mod.graph = mod.graph_module.graph

        # decapsulate method will short circuit the kt_regroup node
        deserialized_model = decapsulate_ir_modules(
            unflatten_ep,
            JsonSerializer,
            short_circuit_pytree_ebc_regroup=True,
            finalize_interpreter_modules=True,
        )

        #  we export the model with ebc1 and unflatten the model,
        #  and then swap with ebc2 (you can think this as the the sharding process
        #  resulting a shardedEBC), so that we can mimic the key-order change
        deserialized_model.sparse.ebc = ebc2

        deserialized_out = deserialized_model(id_list_features)
        for key in eager_out.keys():
            torch.testing.assert_close(deserialized_out[key], eager_out[key])

    def test_cast_in_regroup(self) -> None:
        class Model(nn.Module):
            def __init__(self, ebc, fpebc, regroup):
                super().__init__()
                self.ebc = ebc
                self.fpebc = fpebc
                self.regroup = regroup

            def forward(
                self,
                features: KeyedJaggedTensor,
            ) -> Dict[str, torch.Tensor]:
                kt1 = self.ebc(features)
                kt2 = self.fpebc(features)
                return self.regroup([kt1, kt2])

        tb1_config = EmbeddingBagConfig(
            name="t1",
            embedding_dim=3,
            num_embeddings=10,
            feature_names=["f1", "f2"],
        )
        tb2_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=4,
            num_embeddings=10,
            feature_names=["f3", "f4"],
        )
        tb3_config = EmbeddingBagConfig(
            name="t3",
            embedding_dim=5,
            num_embeddings=10,
            feature_names=["f5"],
        )

        ebc = EmbeddingBagCollection(
            tables=[tb1_config, tb3_config],
            is_weighted=False,
        )
        max_feature_lengths = {"f3": 100, "f4": 100}
        fpebc = FeatureProcessedEmbeddingBagCollection(
            EmbeddingBagCollection(
                tables=[tb2_config],
                is_weighted=True,
            ),
            PositionWeightedModuleCollection(
                max_feature_lengths=max_feature_lengths,
            ),
        )
        data_type = DataType.BF16

        regroup = KTRegroupAsDict(
            [["f1", "f3", "f5"], ["f2", "f4"]], ["odd", "even"], emb_dtype=data_type
        )
        model = Model(ebc, fpebc, regroup)
        self.assertEqual(model.regroup._emb_dtype, data_type)

        id_list_features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f2", "f3", "f4", "f5"],
            values=torch.tensor([0, 1, 2, 3, 2, 3, 4, 5, 6, 7, 8, 9, 1, 1, 2]),
            offsets=torch.tensor([0, 2, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15]),
        )
        # Serialize EBC
        model, sparse_fqns = encapsulate_ir_modules(model, JsonSerializer)
        ep = torch.export.export(
            model,
            (id_list_features,),
            {},
            strict=False,
            # Allows KJT to not be unflattened and run a forward on unflattened EP
            preserve_module_call_signature=(tuple(sparse_fqns)),
        )

        for node in ep.graph.nodes:
            if (
                node.op == "call_function"
                and node.target == torch.ops.torchrec.ir_kt_regroup.default
            ):
                for meta in node.meta["val"]:
                    self.assertEqual(meta.dtype, data_type_to_dtype(data_type))

        unflatten_ep = torch.export.unflatten(ep)
        deserialized = decapsulate_ir_modules(unflatten_ep, JsonSerializer)
        self.assertEqual(deserialized.regroup._emb_dtype, data_type)

    def test_qualname(self) -> None:
        tb1_config = EmbeddingBagConfig(
            name="t1",
            embedding_dim=3,
            num_embeddings=10,
            feature_names=["f1", "f2"],
        )
        tb3_config = EmbeddingBagConfig(
            name="t3",
            embedding_dim=5,
            num_embeddings=10,
            feature_names=["f5"],
        )

        ebc = EmbeddingBagCollection(
            tables=[tb1_config, tb3_config],
            is_weighted=False,
        )

        # for an instance it should return the full qualified class name
        self.assertEqual(
            qualname(ebc), "torchrec.modules.embedding_modules.EmbeddingBagCollection"
        )

        # for a class type it should also return the full qualified class namme
        self.assertEqual(
            qualname(EmbeddingBagCollection),
            "torchrec.modules.embedding_modules.EmbeddingBagCollection",
        )
