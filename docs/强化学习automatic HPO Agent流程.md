Baseline
基本信息
模型：Qwen3-8B Dense
运行配置：单机8卡 C550，单卡280TFlops算力，64G显存， 7×128GB/s的通信带宽
强化学习框架：Verl 0.7版本
初始prompt
模型大小 Qwen3 Dense模型 8B参数量。单机8卡运行，每张卡280TFlops算力，64G显存。使用Verl 0.7框架进行强化学习的RL训练。Verl运行的超参数如下：
python3 -m verl.trainer.main_ppo --config-path=config \
    --config-name='ppo_megatron_trainer.yaml'\
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/lhycpfs/lhy/v5000-rl/data/dapo17k.parquet \
    data.val_files=/mnt/lhycpfs/lhy/v5000-rl/data/aime24.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/mnt/lhycpfs/lhy/model/Qwen3-8B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4 \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1  \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.megatron.param_offload=True \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.sequence_parallel=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_grpo_example_dapo17k' \
    trainer.experiment_name='qwen3_8b_megatron' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.total_training_steps=200 \
    trainer.total_epochs=1 $@ \
    2>&1 | tee -a $logfile;  
    
请综合rollout推理速度，actor模型训练速度，显存和通信开销。给出可以优化的参数空间， 包括已有参数的修改建议和脚本中未开启的参数，并以
参数1: 修改建议及说明
参数2: 修改建议及说明
...
的形式给出
已知强化学习第三步的性能如下：
step:3 - actor/entropy:0.3089402914047241 - perf/mfu/actor_infer:0 - actor/kl_loss:0.0008863400135851407 - actor/kl_coef:0.0010000000000000002 - actor/pg_clipfrac:0.0003315394355922763 - actor/ppo_kl:1.1828059159313398e-05 - actor/pg_clipfrac_lower:0.0 - actor/pg_loss:6.145285442471504e-06 - actor/grad_norm:0.04106206008821267 - perf/mfu/actor:0.41926334277752103 - perf/max_memory_allocated_gb:67.39736652374268 - perf/max_memory_reserved_gb:75.13671875 - perf/cpu_memory_used_gb:363.3620414733887 - actor/lr:1e-06 - training/global_step:3 - training/epoch:0 - critic/score/mean:-0.848437488079071 - critic/score/max:1.0 - critic/score/min:-1.0 - critic/rewards/mean:-0.848437488079071 - critic/rewards/max:1.0 - critic/rewards/min:-1.0 - critic/advantages/mean:-0.00512124365195632 - critic/advantages/max:1.7888524532318115 - critic/advantages/min:-1.7888524532318115 - critic/returns/mean:-0.00512124365195632 - critic/returns/max:1.7888524532318115 - critic/returns/min:-1.7888524532318115 - response_length/mean:4003.34130859375 - response_length/max:4096.0 - response_length/min:1733.0 - response_length/clip_ratio:0.8968750238418579 - response_length_non_aborted/mean:4003.34130859375 - response_length_non_aborted/max:4096.0 - response_length_non_aborted/min:1733.0 - response_length_non_aborted/clip_ratio:0.8968750238418579 - response/aborted_ratio:0.0 - prompt_length/mean:154.83203125 - prompt_length/max:561.0 - prompt_length/min:76.0 - prompt_length/clip_ratio:0.0 - num_turns/min:2 - num_turns/max:2 - num_turns/mean:2.0 - timing_s/start_profile:8.429680019617081e-05 - timing_s/agent_loop/generate_sequences/min:169.55328694637865 - timing_s/agent_loop/generate_sequences/max:690.4329333119094 - timing_s/agent_loop/generate_sequences/mean:496.7219781413325 - timing_s/agent_loop/tool_calls/min:0.0 - timing_s/agent_loop/tool_calls/max:0.0 - timing_s/agent_loop/tool_calls/mean:0.0 - timing_s/agent_loop/slowest/generate_sequences:690.4329333119094 - timing_s/agent_loop/slowest/tool_calls:0.0 - timing_s/agent_loop/slowest/prompt_length:104 - timing_s/agent_loop/slowest/response_length:4096 - timing_s/gen:698.8797343205661 - timing_s/reward:0.00015538185834884644 - timing_s/old_log_prob:143.3216061014682 - timing_s/ref:98.13646441046149 - timing_s/adv:0.033962166868150234 - timing_s/update_actor:327.778207520023 - timing_s/step:1268.4882453735918 - timing_s/stop_profile:4.969164729118347e-05 - timing_per_token_ms/update_actor:0.06158394508406504 - timing_per_token_ms/adv:6.380912981276378e-06 - timing_per_token_ms/ref:0.01843817098374051 - timing_per_token_ms/gen:0.1363860178363828 - perf/total_num_tokens:5322462 - perf/time_per_step:1268.4882453735918 - perf/throughput:524.4886993840887 - perf/tgs/gen:951.9631451995043 - perf/tgs/actor:2029.7497964667414
初始参数说明
通用参数
值
备注
data.train_batch_size
256

data.max_prompt_length
1024

data.max_response_length
4096

data.filter_overlong_prompts
True

Rollout参数（actor_rollout_ref.rollout.）


log_prob_micro_batch_size_per_gpu
64

tensor_model_parallel_size
4

name
vllm

......


第一轮优化建议

修改的参数
值
备注
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 
1->16
减少旧策略的生成时间
actor_rollout_ref.rollout.tensor_model_parallel_size
4->2
减少推理时间
actor_rollout_ref.rollout.n=4
5->4
减少组大小
新增的参数


actor_rollout_ref.model.enable_gradient_checkpointing
True
开梯度累积
actor_rollout_ref.actor.megatron.use_distributed_optimizer
True
Zero 1
actor_rollout_ref.actor.megatron.sequence_parallel
True
开SP
actor_rollout_ref.rollout.enable_chunked_prefill
True
Prefill碎片化减少Decoding延迟
actor_rollout_ref.rollout.max_num_batched_tokens
16384
更好利用continous batching
actor_rollout_ref.rollout.free_cache_engine
True

actor_rollout_ref.rollout.enforce_eager
True

优化后性能对比

初始baseline
优化前性能
step:3 - actor/entropy:0.3089402914047241 - perf/mfu/actor_infer:0 - actor/kl_loss:0.0008863400135851407 - actor/kl_coef:0.0010000000000000002 - actor/pg_clipfrac:0.0003315394355922763 - actor/ppo_kl:1.1828059159313398e-05 - actor/pg_clipfrac_lower:0.0 - actor/pg_loss:6.145285442471504e-06 - actor/grad_norm:0.04106206008821267 - perf/mfu/actor:0.41926334277752103 - perf/max_memory_allocated_gb:67.39736652374268 - perf/max_memory_reserved_gb:75.13671875 - perf/cpu_memory_used_gb:363.3620414733887 - actor/lr:1e-06 - training/global_step:3 - training/epoch:0 - critic/score/mean:-0.848437488079071 - critic/score/max:1.0 - critic/score/min:-1.0 - critic/rewards/mean:-0.848437488079071 - critic/rewards/max:1.0 - critic/rewards/min:-1.0 - critic/advantages/mean:-0.00512124365195632 - critic/advantages/max:1.7888524532318115 - critic/advantages/min:-1.7888524532318115 - critic/returns/mean:-0.00512124365195632 - critic/returns/max:1.7888524532318115 - critic/returns/min:-1.7888524532318115 - response_length/mean:4003.34130859375 - response_length/max:4096.0 - response_length/min:1733.0 - response_length/clip_ratio:0.8968750238418579 - response_length_non_aborted/mean:4003.34130859375 - response_length_non_aborted/max:4096.0 - response_length_non_aborted/min:1733.0 - response_length_non_aborted/clip_ratio:0.8968750238418579 - response/aborted_ratio:0.0 - prompt_length/mean:154.83203125 - prompt_length/max:561.0 - prompt_length/min:76.0 - prompt_length/clip_ratio:0.0 - num_turns/min:2 - num_turns/max:2 - num_turns/mean:2.0 - timing_s/start_profile:8.429680019617081e-05 - timing_s/agent_loop/generate_sequences/min:169.55328694637865 - timing_s/agent_loop/generate_sequences/max:690.4329333119094 - timing_s/agent_loop/generate_sequences/mean:496.7219781413325 - timing_s/agent_loop/tool_calls/min:0.0 - timing_s/agent_loop/tool_calls/max:0.0 - timing_s/agent_loop/tool_calls/mean:0.0 - timing_s/agent_loop/slowest/generate_sequences:690.4329333119094 - timing_s/agent_loop/slowest/tool_calls:0.0 - timing_s/agent_loop/slowest/prompt_length:104 - timing_s/agent_loop/slowest/response_length:4096 - timing_s/gen:698.8797343205661 - timing_s/reward:0.00015538185834884644 - timing_s/old_log_prob:143.3216061014682 - timing_s/ref:98.13646441046149 - timing_s/adv:0.033962166868150234 - timing_s/update_actor:327.778207520023 - timing_s/step:1268.4882453735918 - timing_s/stop_profile:4.969164729118347e-05 - timing_per_token_ms/update_actor:0.06158394508406504 - timing_per_token_ms/adv:6.380912981276378e-06 - timing_per_token_ms/ref:0.01843817098374051 - timing_per_token_ms/gen:0.1363860178363828 - perf/total_num_tokens:5322462 - perf/time_per_step:1268.4882453735918 - perf/throughput:524.4886993840887 - perf/tgs/gen:951.9631451995043 - perf/tgs/actor:2029.7497964667414 - 平均显存占用47.68%，显存峰值占用70.88%
按给出的建议优化后的参数（后覆盖前）：
python3 -m verl.trainer.main_ppo --config-path=config \
    --config-name='ppo_megatron_trainer.yaml' \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/lhycpfs/lhy/v5000-rl/data/dapo17k.parquet \
    data.val_files=/mnt/lhycpfs/lhy/v5000-rl/data/aime24.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/mnt/lhycpfs/lhy/model/Qwen3-8B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4 \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1  \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.megatron.param_offload=True \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.sequence_parallel=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_grpo_example_dapo17k' \
    trainer.experiment_name='qwen3_8b_megatron_optim' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.total_training_steps=200 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16  \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.megatron.use_distributed_optimizer=True \
    actor_rollout_ref.actor.megatron.sequence_parallel=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    trainer.total_epochs=1 $@ \
    2>&1 | tee $logfile;
优化后性能
step:3 - actor/entropy:0.30538251996040344 - perf/mfu/actor_infer:0 - actor/kl_loss:0.0009221725379973122 - actor/kl_coef:0.0010000000000000005 - actor/pg_clipfrac:0.00062446137343386 - actor/ppo_kl:1.33516091400665e-05 - actor/pg_clipfrac_lower:0.0 - actor/pg_loss:1.505733234807849e-05 - actor/grad_norm:0.04788362826607066 - perf/mfu/actor:0.4187941496285902 - perf/max_memory_allocated_gb:77.39132595062256 - perf/max_memory_reserved_gb:86.19921875 - perf/cpu_memory_used_gb:579.793327331543 - actor/lr:1e-06 - training/global_step:3 - training/epoch:0 - critic/score/mean:-0.84765625 - critic/score/max:1.0 - critic/score/min:-1.0 - critic/rewards/mean:-0.84765625 - critic/rewards/max:1.0 - critic/rewards/min:-1.0 - critic/advantages/mean:-0.004950599744915962 - critic/advantages/max:1.4999985694885254 - critic/advantages/min:-1.4999985694885254 - critic/returns/mean:-0.004950599744915962 - critic/returns/max:1.4999985694885254 - critic/returns/min:-1.4999985694885254 - response_length/mean:4004.962890625 - response_length/max:4096.0 - response_length/min:1610.0 - response_length/clip_ratio:0.8955078125 - response_length_non_aborted/mean:4004.962890625 - response_length_non_aborted/max:4096.0 - response_length_non_aborted/min:1610.0 - response_length_non_aborted/clip_ratio:0.8955078125 - response/aborted_ratio:0.0 - prompt_length/mean:154.83203125 - prompt_length/max:561.0 - prompt_length/min:76.0 - prompt_length/clip_ratio:0.0 - num_turns/min:2 - num_turns/max:2 - num_turns/mean:2.0 - timing_s/start_profile:8.477969095110893e-05 - timing_s/agent_loop/generate_sequences/min:82.16073377616704 - timing_s/agent_loop/generate_sequences/max:489.12789867585525 - timing_s/agent_loop/generate_sequences/mean:334.3960995739799 - timing_s/agent_loop/tool_calls/min:0.0 - timing_s/agent_loop/tool_calls/max:0.0 - timing_s/agent_loop/tool_calls/mean:0.0 - timing_s/agent_loop/slowest/generate_sequences:489.12789867585525 - timing_s/agent_loop/slowest/tool_calls:0.0 - timing_s/agent_loop/slowest/prompt_length:101 - timing_s/agent_loop/slowest/response_length:4096 - timing_s/gen:497.47404412087053 - timing_s/reward:0.00014821020886301994 - timing_s/old_log_prob:87.19199371244758 - timing_s/ref:78.85946633201092 - timing_s/adv:0.028186508920043707 - timing_s/update_actor:263.8793420009315 - timing_s/step:927.6805633823387 - timing_s/stop_profile:5.0960108637809753e-05 - timing_per_token_ms/update_actor:0.06194888804918068 - timing_per_token_ms/ref:0.018513219770733825 - timing_per_token_ms/gen:0.12130312052304015 - timing_per_token_ms/adv:6.617126116597852e-06 - perf/total_num_tokens:4259630 - perf/time_per_step:927.6805633823387 - perf/throughput:573.9623864260611 - perf/tgs/gen:1070.3146350900481 - perf/tgs/actor:2017.7924727359689 - 平均显存占用58.23%，显存峰值占用92.23%

请进一步分析参数和显存开销，综合rollout推理速度，actor模型训练速度，显存和通信开销。给出进一步可优化的参数空间， 包括已有参数的修改建议和脚本中未开启的参数，并以
参数1: 修改建议及说明
参数2: 修改建议及说明
...
的形式给出

性能提升与优化反馈
性能指标
旧值
新值
提升/下降比
step time
1268s
927s
↓27%
gen time
699s
497s
↓29%
old_log_prob
143s
87s
↓39%
update_actor
328s
264s
↓20%
throughput
524 tok/s
574 tok/s
↑9.5%
rollout tgs
952
1070
↑12%
训练参数优化建议
值
备注
actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
1->2
显存接近满，MFU上不去，通信瓶颈，TP4产生大量allreduce 通信
actor_rollout_ref.actor.megatron.tensor_model_parallel_size
4->2

actor_rollout_ref.actor.megatron.tp_comm_overlap
None->True
overlap TP通信
actor_rollout_ref.actor.megatron.overlap_grad_reduce
None->True
overlap反向DP all reduce通信
actor_rollout_ref.actor.megatron.grad_reduce_in_fp32
True->False
梯度累积改bf16
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
1->2
gemm太碎
actor_rollout_ref.actor.megatron.optimizer_offload
True->False
影响训练性能
推理参数优化反馈
值
备注
actor_rollout_ref.rollout.gpu_memory_utilization
0.6->0.75

actor_rollout_ref.rollout.max_num_batched_tokens
16384->32768
继续优化 VLLM continuous batching的性能
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu
1->16->32/64
继续优化old prob推理时间
actor_rollout_ref.rollout.max_num_seqs
512
显式增加VLLM batching
Ref Model


actor_rollout_ref.ref.megatron.tensor_model_parallel_size
1->2
不是瓶颈
结果：爆显存爆得很明显，且后续进行显存调整依旧是会连续爆显存
且通过多轮对话发现，大模型并不具备进一步精细规划显存的能力

python3 -m verl.trainer.main_ppo --config-path=config \    --config-name='ppo_megatron_trainer.yaml'\    algorithm.adv_estimator=grpo \    data.train_files=/mnt/lhycpfs/lhy/v5000-rl/data/dapo17k.parquet \    data.val_files=/mnt/lhycpfs/lhy/v5000-rl/data/aime24.parquet \    data.train_batch_size=128 \    data.max_prompt_length=1024 \    data.max_response_length=4096 \    data.filter_overlong_prompts=True \    data.truncation='error' \    actor_rollout_ref.model.path=/mnt/lhycpfs/lhy/model/Qwen3-8B \    actor_rollout_ref.actor.optim.lr=1e-6 \    actor_rollout_ref.actor.use_kl_loss=True \    actor_rollout_ref.actor.kl_loss_coef=0.001 \    actor_rollout_ref.actor.kl_loss_type=low_var_kl \    actor_rollout_ref.actor.entropy_coeff=0 \    actor_rollout_ref.rollout.name=vllm \    algorithm.use_kl_in_reward=False \    trainer.critic_warmup=0 \    trainer.logger=['console','wandb'] \    trainer.project_name='verl_grpo_example_dapo17k' \    trainer.experiment_name='qwen3_8b_megatron_optim_max' \    trainer.n_gpus_per_node=8 \    trainer.nnodes=1 \    trainer.val_before_train=False \    trainer.save_freq=20 \    trainer.test_freq=20 \    actor_rollout_ref.actor.ppo_mini_batch_size=64 \    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4 \    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \    actor_rollout_ref.actor.megatron.sequence_parallel=True \    actor_rollout_ref.actor.megatron.use_distributed_optimizer=True \    actor_rollout_ref.actor.megatron.optimizer_offload=True \    actor_rollout_ref.actor.megatron.param_offload=False \    actor_rollout_ref.model.enable_gradient_checkpointing=True \    actor_rollout_ref.rollout.n=4 \    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=12 \    actor_rollout_ref.rollout.gpu_memory_utilization=0.64 \    actor_rollout_ref.rollout.enable_chunked_prefill=True \    actor_rollout_ref.rollout.max_num_batched_tokens=32768\    actor_rollout_ref.rollout.max_num_seqs=128 \    actor_rollout_ref.rollout.free_cache_engine=True \    actor_rollout_ref.rollout.enforce_eager=False \    actor_rollout_ref.rollout.enable_prefix_caching=True\    actor_rollout_ref.ref.megatron.param_offload=True \    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=4 \    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1 \    actor_rollout_ref.ref.megatron.sequence_parallel=True \    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=12 \    trainer.total_epochs=5 $@ \    2>&1 | tee $logfile;

经过进一步手工多轮精细的调优得到的性能如下：
step:3 - actor/entropy:0.3134717345237732 - perf/mfu/actor_infer:0 - actor/kl_loss:0.0008102378685634903 - actor/kl_coef:0.0010000000000000005 - actor/pg_clipfrac:0.0003905205694536562 - actor/ppo_kl:5.390019401829704e-05 - actor/pg_clipfrac_lower:0.0 - actor/pg_loss:0.0004250430974934716 - actor/grad_norm:0.03962952742918853 - perf/mfu/actor:0.4778220201439828 - perf/max_memory_allocated_gb:80.64591979980469 - perf/max_memory_reserved_gb:85.6328125 - perf/cpu_memory_used_gb:534.672679901123 - actor/lr:1e-06 - training/global_step:3 - training/epoch:0 - critic/score/mean:-0.890625 - critic/score/max:1.0 - critic/score/min:-1.0 - critic/rewards/mean:-0.890625 - critic/rewards/max:1.0 - critic/rewards/min:-1.0 - critic/advantages/mean:-0.0018151035765185952 - critic/advantages/max:1.4999985694885254 - critic/advantages/min:-1.4999985694885254 - critic/returns/mean:-0.0018151035765185952 - critic/returns/max:1.4999985694885254 - critic/returns/min:-1.4999985694885254 - response_length/mean:4020.28125 - response_length/max:4096.0 - response_length/min:1954.0 - response_length/clip_ratio:0.91015625 - response_length_non_aborted/mean:4020.28125 - response_length_non_aborted/max:4096.0 - response_length_non_aborted/min:1954.0 - response_length_non_aborted/clip_ratio:0.91015625 - response/aborted_ratio:0.0 - prompt_length/mean:154.515625 - prompt_length/max:380.0 - prompt_length/min:80.0 - prompt_length/clip_ratio:0.0 - num_turns/min:2 - num_turns/max:2 - num_turns/mean:2.0 - timing_s/start_profile:8.297059684991837e-05 - timing_s/agent_loop/generate_sequences/min:62.73702158872038 - timing_s/agent_loop/generate_sequences/max:239.10129268746823 - timing_s/agent_loop/generate_sequences/mean:198.9852808425585 - timing_s/agent_loop/tool_calls/min:0.0 - timing_s/agent_loop/tool_calls/max:0.0 - timing_s/agent_loop/tool_calls/mean:0.0 - timing_s/agent_loop/slowest/generate_sequences:239.10129268746823 - timing_s/agent_loop/slowest/tool_calls:0.0 - timing_s/agent_loop/slowest/prompt_length:136 - timing_s/agent_loop/slowest/response_length:4096 - timing_s/gen:242.7217347882688 - timing_s/reward:0.00016184989362955093 - timing_s/old_log_prob:41.48836747743189 - timing_s/ref:40.86255992203951 - timing_s/adv:0.014681442640721798 - timing_s/update_actor:114.99425340164453 - timing_s/step:440.19567225221545 - timing_s/stop_profile:5.030538886785507e-05 - timing_per_token_ms/ref:0.019117022872575905 - timing_per_token_ms/adv:6.868524030324173e-06 - timing_per_token_ms/update_actor:0.05379858179928502 - timing_per_token_ms/gen:0.11791858797399747 - perf/total_num_tokens:2137496 - perf/time_per_step:440.19567225221545 - perf/throughput:606.9732549458414 - perf/tgs/gen:1100.7955271622986 - perf/tgs/actor:2323.481322729984
相比baseline性能提升（batch_size 256->128），可以通过修改epoch拉平单prompt backward次数，但是1/step time依旧提升了44.1%

GRPO与Verl的调参
GRPO过程
actor old model（本轮冻结）, ref model（永久冻结）, actor new model（一直在更新）
1. actor old做rollout生成一组rollout，同时生成rollout的概率old_probs
2. ref model生成old_probs_ref
3. reward对rollout结果进行打分，计算组内的均值和方差得到每条样本的组内advantage A
4. 可训练的actor根据prompt+rollout结果，重新前向传播，生成new_probs
5. 得到比值r = exp(log(new_prob)-log(old_probs))
6. 计算GRPO loss = PPO policy loss - KL loss。
第一部分PPO policy loss = -min(A*r+clip(A*r, 1-e, 1+e))
第二部分KL loss = KL(new_probs, old_probs_ref)
7. 利用GRPO loss backward+optimizer step更新模型。然后回到第4步，直到完成本次rollout所有的epochs才重新开始1
1-3属于推理过程，actor old model以及KV cache算完1后释放，ref model算完2后offload。4-7属于训练过程，actor model始终inflight并被更新，2和4属于teaching force forward process。
1-3后需要缓存的只有old probs, ref probs, prompt+rollout tokens（5120*4*128 = 2621440 tokens，显存不大），optimizer steps = bsz/ppo_bsz * epoch
但是verl里需要指定actor_rollout_ref.rollout.gpu_memory_utilization为VLLM的KV-cache预留显存，这是一个毒点。显存峰值一般存在于计算old probs的时候，此时VLLM的显存没有释放，old actor也没释放，算old probs同时存在的时候， old probs和ref probs为了提高计算效率batch_size不能太小，这里设置为12基本接近最大了，16就会爆。
训练参数优化
值
备注
actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
1
TP+SP的方式对显存开销的节省更明显
actor_rollout_ref.actor.megatron.tensor_model_parallel_size
4

actor_rollout_ref.actor.megatron.sequence_parallel
True

actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
2
不能开1防止gemm效率低
actor_rollout_ref.actor.megatron.optimizer_offload
True
优化器状态很大，offload的影响不算大，但是也存疑，因为理论上训练阶段推理显存全部可以释放
actor_rollout_ref.actor.megatron.param_offload
False
不能开，开了对性能影响很大
actor_rollout_ref.model.enable_gradient_checkpointing=True
存疑，之前actor大切分频繁爆显存
如果完美内存回收的话理论上不用开，因为如果VLLM释放后，模型权重其实比较小
推理参数优化
值
备注
actor_rollout_ref.rollout.gpu_memory_utilization
0.64
VLLM的KV-cache预留显存，这个值目前看来有毒点，和预期不一致，训练阶段理论可以完全释放推理阶段用到的所有
actor_rollout_ref.rollout.max_num_batched_tokens
16384->32768
优化 VLLM continuous batching的性能
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu
16->12
继续优化old prob推理时间
actor_rollout_ref.rollout.max_num_seqs
128
本质无影响，受VLLM最大batch tokens影响
actor_rollout_ref.rollout.free_cache_engine
True
必须开，释放KV-cache的显存
actor_rollout_ref.rollout.force_eager
False/True
False吞吐高，True关闭torch.compile，顺序图执行，减少显存碎片，优化峰值显存



actor_rollout_ref.ref.megatron.tensor_model_parallel_size
2->4 
需要和actor model保持一致，graph效率更高
actor_rollout_ref.ref.megatron.sequence_parallel
True
开SP
actor_rollout_ref.ref.megatron.param_offload
False
必须开，ref model不需要长时间放在显存里
思路：
寻找分别对old actor和new actor做不同并行切分策略的方法
验证new actor是否真的只能用小切分的显存（受到VLLM预留最大显存限制），关系到能否开更高的并行，去掉gc以及optimizer offload
探寻更高效的显存回收机制（框架调优）


8卡*64G显存的配置，目前verl训练Qwen3 8B的脚本参数。必须固定max_response_length和rollout.n，当前训练峰值显存出现在rollout（87.92%），actore训练阶段（73.59%），还有哪些参数可以优化。参数如下这套并行配置在64G显存，8卡下如何进一步优化推理吞吐和训练MFU
python3 -m verl.trainer.main_ppo --config-path=config \
    --config-name='ppo_megatron_trainer.yaml'\
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/lhycpfs/lhy/v5000-rl/data/dapo17k.parquet \
    data.val_files=/mnt/lhycpfs/lhy/v5000-rl/data/aime24.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/mnt/lhycpfs/lhy/model/Qwen3-8B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.name=vllm \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_grpo_example_dapo17k' \
    trainer.experiment_name='qwen3_8b_megatron_optim_max' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.sequence_parallel=True \
    actor_rollout_ref.actor.megatron.use_distributed_optimizer=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    actor_rollout_ref.actor.megatron.param_offload=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=12 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.64 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768\
    actor_rollout_ref.rollout.max_num_seqs=128 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.ref.megatron.param_offload=True \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=4 \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.sequence_parallel=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=12 \
    trainer.total_epochs=5 $@ \
    2>&1 | tee $logfile;

actor/entropy:0.3140407204627991 - perf/mfu/actor_infer:0 - actor/kl_loss:0.0007927680430839246 - actor/kl_coef:0.0010000000000000005 - actor/pg_clipfrac:0.00038341219101312163 - actor/ppo_kl:6.199137132512078e-07 - actor/pg_clipfrac_lower:0.0 - actor/pg_loss:0.00024438369655399583 - actor/grad_norm:0.039331942690620056 - perf/mfu/actor:0.48026394402806283 - perf/max_memory_allocated_gb:80.60690259933472 - perf/max_memory_reserved_gb:90.751953125 - perf/cpu_memory_used_gb:420.9437789916992 - actor/lr:1e-06 - training/global_step:3 - training/epoch:0 - critic/score/mean:-0.88671875 - critic/score/max:1.0 - critic/score/min:-1.0 - critic/rewards/mean:-0.88671875 - critic/rewards/max:1.0 - critic/rewards/min:-1.0 - critic/advantages/mean:-0.0028102584183216095 - critic/advantages/max:1.4999985694885254 - critic/advantages/min:-1.4999985694885254 - critic/returns/mean:-0.0028102584183216095 - critic/returns/max:1.4999985694885254 - critic/returns/min:-1.4999985694885254 - response_length/mean:4015.3984375 - response_length/max:4096.0 - response_length/min:1757.0 - response_length/clip_ratio:0.900390625 - response_length_non_aborted/mean:4015.3984375 - response_length_non_aborted/max:4096.0 - response_length_non_aborted/min:1757.0 - response_length_non_aborted/clip_ratio:0.900390625 - response/aborted_ratio:0.0 - prompt_length/mean:154.515625 - prompt_length/max:380.0 - prompt_length/min:80.0 - prompt_length/clip_ratio:0.0 - num_turns/min:2 - num_turns/max:2 - num_turns/mean:2.0 - timing_s/start_profile:8.176453411579132e-05 - timing_s/agent_loop/generate_sequences/min:56.121871799230576 - timing_s/agent_loop/generate_sequences/max:237.11041424795985 - timing_s/agent_loop/generate_sequences/mean:197.1275855792701 - timing_s/agent_loop/tool_calls/min:0.0 - timing_s/agent_loop/tool_calls/max:0.0 - timing_s/agent_loop/tool_calls/mean:0.0 - timing_s/agent_loop/slowest/generate_sequences:237.11041424795985 - timing_s/agent_loop/slowest/tool_calls:0.0 - timing_s/agent_loop/slowest/prompt_length:150 - timing_s/agent_loop/slowest/response_length:4096 - timing_s/gen:240.5041181501001 - timing_s/reward:0.0001397412270307541 - timing_s/old_log_prob:41.35555414296687 - timing_s/ref:40.63676842302084 - timing_s/adv:0.015331225469708443 - timing_s/update_actor:114.17891584523022 - timing_s/step:436.80501881986856 - timing_s/stop_profile:4.723668098449707e-05 - timing_per_token_ms/adv:7.1809153130537215e-06 - timing_per_token_ms/ref:0.019033650846662403 - timing_per_token_ms/gen:0.11698331138823986 - timing_per_token_ms/update_actor:0.05347968607211921 - perf/total_num_tokens:2134996 - perf/time_per_step:436.80501881986856 - perf/throughput:610.9693993924891 - perf/tgs/gen:1109.6462798755153 - perf/tgs/actor:2337.3360836754573