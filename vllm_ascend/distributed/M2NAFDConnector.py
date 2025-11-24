from dataclasses import dataclass
from typing import Any

from vllm.distributed.afd_transfer.afd_connector import (AFDConnectorBase, AFDConnectorFactory,
                            AFDConnectorMetadata)


__all__ = ["AFDConnectorBase", "AFDConnectorMetadata", "AFDConnectorFactory"]

import torch_npu
import torch

from torch.distributed.distributed_c10d import _get_default_group
from vllm.distributed.parallel_state import init_afd_process_group, DefaultProcessGroupSwitcher
import re

import torch
from torch.distributed.distributed_c10d import  _update_default_pg, _get_default_group

from vllm.distributed.parallel_state import init_afd_process_group, init_model_parallel_group
from vllm.distributed.afd_transfer.afd_connector.metadata import (
    M2NAFDConnectorMetadata)

from vllm.logger import init_logger
from vllm.config import VllmConfig
logger = init_logger(__name__)


# # TODO(yxj):move to ascend ,use kwargs 
# @dataclass
# class M2NAFDConnectorMetadata:
#     def __init__(self):
#         self.topk_idx = None
#         self.topk_weights = None
#         self.moe_expert_num = 0
#         self.scale = None
#         self.handle = None
#         self.quant_mode = 0
#         self.aiv_num = 0
#         self.batch_size = 0
#         self.h = 0
#         self.k = 0
#         self.expert_token_nums_type = 0
#         self.expand_x_type = torch.float16

class M2NAFDConnector(AFDConnectorBase):
    def __init__(self, 
        rank: int, 
        local_rank: int,
        config: "VllmConfig"
)-> None:
        self.rank = rank
        self.local_rank = local_rank
        self._initialized = False
        self.config = config
        self.attn_size = 0
        self.ffn_size = 0
    
    def close(self) -> None:
        """Close the connector and release resources."""
        # destroy process group
        pass
    
    def init_afd_connector(self) -> None:
        """Initialize the AFD connector."""
        afd_size = self.config.afd_config.afd_extra_config.get("afd_size")
        role = self.config.afd_config.afd_role
        self.attn_size, self.ffn_size = map(
            int,
            re.match(r"(\d+)\D+(\d+)", afd_size).groups())
        #ffn_ranks = [i for i in range(ffn_size, ffn_size + attn_size)]
        #attn_ranks = [i for i in range(attn_size)]
        world_rank = self.rank if role == "attention" else self.rank + self.attn_size
        # p2p_rank = self.rank if role == "attention" else self.rank + self.ffn_size
        self.rank = world_rank
        logger.info(
            f"world_size = {self.ffn_size + self.attn_size}, world_rank = {world_rank}")
        # TODO : get backend to replace hardcode
        self.afd_pg = init_afd_process_group(
            backend="hccl",
            init_method=f"tcp://127.0.0.1:11039",
            world_size=self.ffn_size + self.attn_size,
            rank=world_rank,
            group_name="afd"
        )

        # if self.rank < self.ffn_size or self.rank >= self.attn_size:

        # p2p_pg = init_afd_process_group(
        #     backend="hccl",
        #     init_method=f"tcp://127.0.0.1:11040",
        #     world_size=self.ffn_size * 2,
        #     rank=p2p_rank,
        #     group_name="p2p"
        # )

        # ffn_ranks = [i for i in range(self.ffn_size, self.ffn_size * 2)]
        # attn_ranks = [i for i in range(self.ffn_size)]

        default_pg_switcher = DefaultProcessGroupSwitcher(
            _get_default_group(), self.afd_pg)
        with default_pg_switcher:
            # sub_group_ranks = []
            # for i in range(len(ffn_ranks)):
            #     ranks = list([attn_ranks[i], ffn_ranks[i]])
            #     sub_group_ranks.append(ranks)
            print(sub_group_ranks)
            self.process_group = init_model_parallel_group([[0,1,2]],
                                                world_rank,
                                                backend="hccl",
                                                group_name="ae")

        logger.info("m2n connector initialized")

        self._initialized = True
    
    def is_initialized(self) -> bool:
        """Check if the connector is initialized and ready to use.
        
        Returns:
            bool: True if the connector is initialized, False otherwise.
        """
        return self._initialized
                                  
    # ATTN发给MOE（ATTN发送）
    # TODO:metadata的获取，最好从框架侧去拿
    def send_attn_output(self, 
                         hidden_states: torch.Tensor,  
                         topk_weights: torch.Tensor, 
                         topk_ids:torch.Tensor, 
                         metadata: AFDConnectorMetadata) -> Any:

        # print('send hidden_states!!!!!!!!!!!!!!!!!!!!!!\n')
        # print(hidden_states)
        # print('send topk_weights!!!!!!!!!!!!!!!!!!!!!!\n')
        # print(topk_weights)
        # print('send topk_ids!!!!!!!!!!!!!!!!!!!!!!\n')
        # print(topk_ids)
        # print('send world_size!!!!!!!!!!!!!!!!!!!!!!\n')
        # print(self.attn_size + self.ffn_size)
        # print('send moe_world_size!!!!!!!!!!!!!!!!!!!!!!\n')
        # print(self.ffn_size)
        # print('send ep_rank_id!!!!!!!!!!!!!!!!!!!!!!\n')
        # print(self.rank)
        # print('send moe_expert_num!!!!!!!!!!!!!!!!!!!!!!\n')
        # print(metadata.m2n_afdconnector_data.moe_expert_num)
        
        if self.rank < self.ffn_size:
            # TODO():move to support aclgraph
            dst = (self.process_group.rank_in_group + 1) % self.process_group.world_size
            print(f'send_attn_output dst is {dst}')
            self.process_group.send_object(metadata,dst)
            print(f'send_attn_output metadata success')
        dynamic_scales = metadata.m2n_afdconnector_data.scale
        # moe_expert_num
        moe_expert_num = metadata.m2n_afdconnector_data.moe_expert_num
        quant_mode = metadata.m2n_afdconnector_data.quant_mode
        aiv_num = metadata.m2n_afdconnector_data.aiv_num
        
        if dynamic_scales is None:
            dynamic_scales = torch.tensor([], dtype=torch.float32, device='npu')
        recv_counts = torch_npu.npu_m2n_distribute_send(x=hidden_states,
                                                        expert_ids=topk_ids,
                                                        expert_scales=topk_weights,
                                                        group_ep=self.afd_pg._get_backend(torch.device("npu")).get_hccl_comm_name(self.rank),
                                                        world_size=self.attn_size + self.ffn_size,
                                                        moe_world_size=self.ffn_size,
                                                        ep_rank_id=self.rank,
                                                        moe_expert_num=moe_expert_num,
                                                        quant_mode=quant_mode,
                                                        aiv_num=aiv_num,
                                                        server_rank_size = 1,
                                                        dynamic_scales=dynamic_scales)
        
        
        
        return recv_counts

    # MOE发给ATTN（ATTN接收）
    def recv_ffn_output(self, hidden_states: torch.Tensor, metadata: AFDConnectorMetadata) -> torch.Tensor:
        # handle = send_attn_output（）recv_counts
        handle = metadata.m2n_afdconnector_data.handle
        moe_expert_num = metadata.m2n_afdconnector_data.moe_expert_num
        aiv_num = metadata.m2n_afdconnector_data.aiv_num

        # print("recv_ffn output handle")
        # print(handle)
        # print("recv_ffn output world_size")
        # print(self.attn_size + self.ffn_size)
        # print("recv_ffn output moe_world_size")
        # print(self.ffn_size)
        # print("recv_ffn output rank")
        # print(self.rank)
        # print("recv_ffn output moe_expert_num")
        # print(moe_expert_num)
        # print("recv_ffn output aiv_num")
        # print(aiv_num)
        
        xOut = torch_npu.npu_n2m_distribute_recv(x=hidden_states,
                                                ep_recv_counts=handle,
                                                group_ep=self.afd_pg._get_backend(torch.device("npu")).get_hccl_comm_name(self.rank),
                                                world_size=self.attn_size + self.ffn_size,
                                                moe_world_size=self.ffn_size,
                                                ep_rank_id=self.rank,
                                                moe_expert_num=moe_expert_num,
                                                server_rank_size = 1,
                                                aiv_num=aiv_num)
        # print("recv_ffn output xOut")
        # print(xOut)
        return xOut
    
    # MOE发给ATTN(MOE发送) 
    def send_ffn_output(self, ffn_output: torch.Tensor, metadata: M2NAFDConnectorMetadata):
        # 配置
        batch_size = metadata.batch_size
        topk_weights = metadata.topk_weights
        moe_expert_num = metadata.moe_expert_num
        aiv_num = metadata.aiv_num
        k = metadata.k
        handle = metadata.handle
        
        # print("send ffn output handle")
        # print(handle)
        # print("send ffn output topk_weights")
        # print(topk_weights)
        # print("send ffn output world_size")
        # print(self.attn_size + self.ffn_size)
        # print("send ffn output moe_world_size")
        # print(self.ffn_size)
        # print("send ffn output rank")
        # print(self.rank)
        # print("send ffn output moe_expert_num")
        # print(moe_expert_num)
        # print("send ffn output k")
        # print(k)
        # print("send ffn output aiv_num")
        # print(aiv_num)
        # print("send ffn output batch_size")
        # print(batch_size)
        # print("send ffn output ffn_output")
        # print(ffn_output)

        torch_npu.npu_n2m_distribute_send(expandX=ffn_output,
                                        ep_send_counts=handle,
                                        expert_scales=topk_weights,
                                        group_ep=self.afd_pg._get_backend(torch.device("npu")).get_hccl_comm_name(self.rank),
                                        world_size=self.attn_size + self.ffn_size,
                                        moe_world_size=self.ffn_size,
                                        ep_rank_id=self.rank,
                                        moe_expert_num=moe_expert_num,# config
                                        batch_size=batch_size,# config
                                        k=k,# config
                                        server_rank_size = 1,
                                        aiv_num=aiv_num)# config 未分核48 
        print(f'send_ffn_output success')
        return
    
    # ATTN发给MOE(MOE接收)
    def recv_attn_output(self, metadata: M2NAFDConnectorMetadata) -> Any: 
        
        print(f'before recv_attn_output metadata is {metadata}') 
        src = (self.process_group.rank_in_group - 1) % self.process_group.world_size
        afdConnectorMetadata = self.process_group.recv_object(src)
        print(f'recv_attn_output metadata success')
        print(f'after recv_attn_output metadata is {metadata}') 
        # TODO(yxj): 对比
        x_type = torch.int8
        if metadata.quant_mode == 0 :
            x_type = metadata.expand_x_type
        batch_size = metadata.batch_size
        quant_mode = metadata.quant_mode
        moe_expert_num = metadata.moe_expert_num
        aiv_num = metadata.aiv_num
        k = metadata.k
        h = metadata.h
        expert_token_nums_type = metadata.expert_token_nums_type

        #npu::npu_m2n_distribute_recv(Tensor x, str group_ep, int world_size, int server_rank_size, int moe_world_size, int ep_rank_id, int moe_expert_num, int quant_mode, int batch_size, int h, int k, int expert_token_nums_type, int aiv_num) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
        expand_x, dynamic_scales, expert_token_nums, recv_counts, expand_scales = torch_npu.npu_m2n_distribute_recv(x = torch.tensor([], dtype=x_type, device='npu'),
                                                                                group_ep=self.afd_pg._get_backend(torch.device("npu")).get_hccl_comm_name(self.rank),
                                                                                world_size=self.attn_size + self.ffn_size,
                                                                                moe_world_size=self.ffn_size,
                                                                                ep_rank_id=self.rank,
                                                                                moe_expert_num=moe_expert_num,
                                                                                quant_mode=quant_mode,
                                                                                batch_size=batch_size,
                                                                                h=h,
                                                                                k=k,
                                                                                expert_token_nums_type=expert_token_nums_type,
                                                                                server_rank_size = 1,
                                                                                aiv_num=aiv_num)
        
        # token_num = expert_token_nums[-1]
        # expand_x = expand_x[:token_num]
        # expand_scales = expand_scales[:token_num]

        # print('recv moe_expert_num!!!!!!!!!!!!!!!!!!!!!!\n')
        # print(moe_expert_num)
        # print('recv world_size!!!!!!!!!!!!!!!!!!!!!!\n')
        # print(self.attn_size + self.ffn_size)
        # print('recv moe_world_size!!!!!!!!!!!!!!!!!!!!!!\n')
        # print(self.ffn_size)
        # print('recv ep_rank_id!!!!!!!!!!!!!!!!!!!!!!\n')
        # print(self.rank)
        # print('recv recv_counts!!!!!!!!!!!!!!!!!!!!!!\n')
        # print(recv_counts[:128])
        # print('recv expert_token_nums!!!!!!!!!!!!!!!!!!!!!!\n')
        # print(expert_token_nums)
        # print('recv topk_weights!!!!!!!!!!!!!!!!!!!!!!\n')
        # print(expand_scales)
        # print('recv hidden_states!!!!!!!!!!!!!!!!!!!!!!\n')
        # print(expand_x)
        
        # print("recv attn output handle")
        # print(recv_counts)
        
        # recv_counts 返程路由
        return expand_x, dynamic_scales, expert_token_nums, recv_counts, expand_scales, None