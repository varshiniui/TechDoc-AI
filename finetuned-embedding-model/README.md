---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:149
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: How can I optimize a kernel that is DRAM bandwidth bound?
  sentences:
  - "2. Saturate memory bandwidth. • Need to hide the corresponding latencies to achieve\
    \ this. • Compute latencies. • Memory access latencies. • Latencies can be hidden\
    \ by having more instructions in flight. Concurrency = Bandwidth x Latency = \n\
    8 x 24 operations in-flight\nFP32\nFP32\nFP32\nFP32\nFP32\nFP32\nFP32\nFP32\n\
    FP32 Latency = 24 cycles\n8 FP32 ops per cycle\n\n23\nHiding Latencies\nIncreasing\
    \ in-flight instructions\n• Two ways to increase in-flight instructions:\n1. Improve\
    \ Instruction-Level Parallelism (ILP). • Higher ILP -> more independent instructions\
    \ per thread. 2. Improve Thread-Level Parallelism (TLP). • Higher TLP -> more\
    \ threads -> more independent \ninstructions per kernel."
  - '• This kernel is DRAM bandwidth bound. 86

    Summary


    87

    Which optimizations to focus on? Solving the bottlenecks

    • Compute bound

    • Reduce instruction count. • E.g., use vector loads/stores. • Use tensor cores.
    • Use lower precision arithmetic, fast math intrinsics. • Bandwidth bound

    • Reduce the amount of data transferred. • Optimize memory access patterns. •
    Lower precision datatypes. • Kernel fusion. • Latency bound

    • Increase number of instructions and memory accesses in-flight. • Increase parallelism,
    occupancy.'
  - "• Shared memory can be useful for:\n• Storing frequently used data\n• Improving\
    \ global memory access patterns\n• Data layout conversion\n• Communication among\
    \ threads of a thread block\n\n62\nShared Memory\nCapacity:\n• Default 48 KiB\
    \ per thread block, opt-in to get more using cudaFuncSetAttribute() with the attribute\
    \ \ncudaFuncAttributeMaxDynamicSharedMemorySize. • Up to 227KiB per thread block\
    \ on Hopper. Organization:\n• Divided into 32 banks, each 4-byte wide. • Successive\
    \ 4-byte words map to successive banks. • Bank index calculation examples:\n•\
    \ (4-byte word index) % 32\n• (1-byte word index / 4) % 32\nPerformance:\n• Slower\
    \ than registers, but much faster than global memory."
- source_sentence: What limits the occupancy of a CUDA kernel?
  sentences:
  - "NVIDIA Ada marked the \ntipping point where ray tracing and neural graphics became\
    \ mainstream. Figure 2. The Age of Neural Rendering Has Arrived - Significant\
    \ AI TOPS \nIncrease Per Frame  \nImage quality has been increasing faster than\
    \ Moore's Law by using neural rendering, and such AI \nrendering techniques will\
    \ continue to expand. DLSS has increased frame rates dramatically by \ngenerating\
    \ a vast majority of pixels at a fraction of the cost of \nnative rendering. DLSS\
    \ -RR (Ray \nReconstruction) has allowed for realistic lighting using path tracing\
    \ by drastically reducing the \nnumber of rays that need to be cast and shaded.\
    \ Blackwell introduces DLSS 4 with multi\n-frame generation that further increases\
    \ \nreal time 3D \nperformance while reducing latency."
  - "• Device (depends on compute capability of the GPU)\n• Achievable (depends on\
    \ kernel implementation + compiler) \n• Achieved (depends mostly on the grid size)\n\
    • Occupancy of a CUDA kernel may be limited by:\n• Register usage\n• SM registers\
    \ are partitioned among threads. • Shared memory usage\n• SM shared memory is\
    \ partitioned among thread blocks. • Thread block size\n• Threads are allocated\
    \ at thread block granularity. \U0001D476\U0001D484\U0001D484\U0001D496\U0001D491\
    \U0001D482\U0001D48F\U0001D484\U0001D49A= \U0001D468\U0001D484\U0001D489\U0001D48A\
    \U0001D486\U0001D497\U0001D482\U0001D483\U0001D48D\U0001D486 # \U0001D482\U0001D484\
    \U0001D495\U0001D48A\U0001D497\U0001D486 \U0001D498\U0001D482\U0001D493\U0001D491\
    \U0001D494 \U0001D491\U0001D486\U0001D493 \U0001D47A\U0001D474\n\U0001D46B\U0001D486\
    \U0001D497\U0001D48A\U0001D484\U0001D486 # \U0001D482\U0001D484\U0001D495\U0001D48A\
    \U0001D497\U0001D486 \U0001D498\U0001D482\U0001D493\U0001D491\U0001D494 \U0001D491\
    \U0001D486\U0001D493 \U0001D47A\U0001D474\nAnalyze the \noccupancy of CUDA \n\
    kernels with NVIDIA \nNsight Compute!"
  - "Detailed specs will be provided for RTX PRO \n6000 Blackwell Workstation Edition\
    \ and RTX PRO 6000 Blackwell Max\n-Q Edition in this paper . Introduction \n \n\
    5 \n \nFigure 1. RTX Blackwell Design Goals \n \nThe following Key Features are\
    \ included in the NVIDIA RTX Blackwell architecture, and will be \ndescribed in\
    \ more detail in the sections below:\n \n● New SM features built for Neural Shading\
    \ \n- New RT Core and Tensor Core features \ndescribed below enhance and accelerate\
    \ neural rendering capabilities. The NVIDIA RTX \nBlackwell SM provides a doubling\
    \ of integer math throughput per clock cycle compared to \nNVIDIA Ada GPUs, which\
    \ can increase the perfor mance of address generation workloads \nthat are crucial\
    \ for neural shading."
- source_sentence: What is the naive implementation for finding the maximum element
    of an array using atomics?
  sentences:
  - "In the case of the deepest sleep state, Blackwell is 10x faster to enter sleep\
    \ than Ada, enabling \nmuch more power savings in the lowest-power sleep state.\
    \ 33 \n \nFigure 22. Real-life Example of Running Inference on SLMs on Ada and\
    \ Blackwell \nIn a real-life example, like running inference on small language\
    \ models as shown in Figure 22 \nabove, power savings of up to 50% can be observed\
    \ through a combination of Blackwell \nperformance (reduced active period), lower\
    \ power transitional states through power and voltage \ngating, and entering the\
    \ deepest sleep state 10x faster than before. DLSS 4 \n \n34 \nDLSS 4 \nDLSS is\
    \ a revolutionary suite of neural rendering technologies that uses AI to boost\
    \ FPS, reduce \nlatency, and improve image quality."
  - "• Strive to avoid bank conflicts. • Use vectorized loads/stores. 75\nAtomics\n\
    \n76\nSerialized! Least efficient access pattern. Most efficient access pattern.\
    \ Using Atomics Efficiently\nAccess Patterns\nSame address\nCoalesced\nScattered\n\
    \n77\n77\nUsing Atomics Efficiently\nExample #1: find the maximum value of an\
    \ array\n• Problem description: given an input array, find the \nmaximum element\
    \ in the array. • Naïve implementation: every thread find its local \nmaximum\
    \ and then atomically updates the global \nmaximum. • N / elements_per_thread\
    \ same-address global atomics."
  - "GDDR7 Memory Subsystem  \nWith Blackwell, NVIDIA is launching GDDR7, a new ultra\
    \ -low voltage GDDR memory standard that \nuses PAM3 (Pulse Amplitude Modulation\n\
    : 3 levels) signaling technology, enabling a substantial \nadvancement in high\
    \ -speed memory design. NVIDIA’s work with the JEDEC technology \nassociation,\
    \ the global leader in the development of standards for the microelectronics industry\
    \ \nwith over 360 member companies, helped to make PAM3 the foundational high\n\
    -frequency \nsignaling technology for GDDR7 DRAM. The RTX PRO 6000 Blackwell Workstation\
    \ Edition and the RTX PRO 6000 Blackwell Max\n-Q \nWorkstation Edition both ship\
    \ with 96GB of 28 Gbps GDDR7 memory and delivers 1.792 TB/sec \npeak memory bandwidth\
    \ ."
- source_sentence: How does Mega Geometry improve real-time ray tracing performance?
  sentences:
  - "GDDR7 is a new ultra -low voltage \nGDDR memory standard that uses PAM3 (Pulse\
    \ Amplitude Modulation) signaling \ntechnology, enabling higher -speed memory\
    \ subsystems and improvements in energy \nefficien cy. ● New 4th Generation RT\
    \ Cores - Significant improvements to the RT Core architecture \nwere made in\
    \ Blackwell, enabling new ray tracing experiences and neural rendering \ntechniques.\
    \ ● New 5th Generation Tensor Cores \n- Includes new FP4 capabilities that can\
    \ double AI \nthroughput while halving the memory requirements. Support for the\
    \ new Second -\nGeneration FP8 Transformer Engine used in our datacenter\n-class\
    \ Blackwell GPUs is also \nincluded. Introduction \n \n6 \n● RTX Mega Geometry\
    \ - A new RTX technology aimed at dramatically increasing the \ngeometric detail\
    \ that is possible in ray\n-traced applications."
  - "• Reports static shared memory usage per thread block. • Hopper has 228 KiB of\
    \ shared memory. • 1KiB per thread block is reserved for system use. • With opt-in\
    \ using dynamic shared memory. • Example:\n• Kernel uses 17408 bytes of shared\
    \ memory per 128-thread block. • Blocks per SM = 233472 / (17408 +1024 ) = 12.66\
    \ \n• Achievable active warps per SM = 12 * 128 / 32 = 48\n• Occupancy = 48 /\
    \ 64 * 100 = 75%\n• Hopper supports up to 64 warps per SM. 40\nOccupancy Limiters\n\
    Thread block size\n• Thread block size is a multiple of warp size (32). • Even\
    \ if you request fewer threads, HW rounds up. • Each thread block can have a maximum\
    \ size of 1024. • Each SM can have up to 64 warps, 32 blocks and 2048 threads\
    \ (Hopper)."
  - "Mega Geometry makes it possible for the application to map \nits tessellation\
    \ process directly to cluster generation, and construct BVHs from the resulting\
    \ \nCLASes extremely quickly. This method unlocks unprecedented real-time performance\
    \ for ray \ntracing of animated displaced subdivision surfaces. Mega Geometry\
    \ API and Architecture Support \nThe functionality surrounding the management\
    \ of BVHs is a fundamental pillar of any ray tracing \nsystem. Mega Geometry is\
    \ a core technology that takes BVH capabilities to the next level, and \n\n \n\
    \ \n \n21 \nempowers applications to invent more creative and efficient geometry\
    \ pipelines than ever."
- source_sentence: How many ways does CUDA expose thread scopes to the programmer?
  sentences:
  - "CUDA/Software\nThread Block\nThread Block\nGrid\nThread Block\nThread Block\n\
    Thread Block\nShared Memory\nGlobal Memory\nCPU\nGPU 0\nGPU 1\n\n18\nThread Scopes\n\
    • To account for non-uniform thread synchronization \ncosts, CUDA has introduced\
    \ the notion of thread scopes. • A thread scope specifies which threads can \n\
    communicate with each other using a primitive such as \nan atomic or a barrier.\
    \ • Thread scopes are exposed to the programmer in 3 ways:\n• PTX\n• CUDA Math\
    \ API\n• CUDA C++ \n• Always use the narrowest scope that ensures correctness\
    \ \nof your application. • More on thread scopes in the GTC session [S62192]:\
    \ \n“Advanced Performance Optimization in CUDA”."
  - "• Visible by all threads in a grid. • Slowest access. Memory Hierarchy\nL2\n\
    DRAM\nHardware\nCUDA/Software\nThread\nThread Block\nThread Block\nThread Block\n\
    Grid\nThread Block\nThread Block\nThread Block\nThread Block\nThread Block\nThread\
    \ Block\nRegisters\nShared \nMemory\nGlobal Memory\nSM\nShared/L1\nRegisters\n\
    Local Memory\n\n16\nSynchronization\nBarriers\n• Grid boundary. • Kernel completion.\
    \ • grid_group::sync() via Cooperative Groups API\n• Requires the kernel to be\
    \ launched via the \ncudaLaunchCooperativeKernel() API\n• Slow! Avoid unless necessary.\
    \ • Thread-block boundary. • __syncthreads() \n• thread_block::sync() via Cooperative\
    \ Groups API\n• Fast! The most common synchronization level. • Warp or sub-warp\
    \ boundary. • __syncwarp()\n• coalesced_group::sync() via Cooperative Groups API\n\
    • Very fast!"
  - "• Thread blocks in a cluster are guaranteed to be concurrently scheduled and\
    \ enable efficient cooperation and data \nsharing for threads across multiple\
    \ SMs. • For more information on this topic visit GTC session [S62192]: “Advanced\
    \ Performance Optimization in CUDA”. Thread Block\nThread Block\nThread Block\n\
    Thread Block\nThread Block\nThread Block\nThread Block\nThread Block\nThread Block\n\
    Thread Block\nGrid\nThread Block\nThread Block\nThread Block\nThread Block\nThread\
    \ Block\nThread Block\nThread Block Cluster\nThread Block Cluster\n\n12\nThread\
    \ Hierarchy\nWhat about warps? • At runtime, a block of threads is divided into\
    \ warps for SIMT execution. • The way a block is partitioned into warps is always\
    \ the same. • Each warp contains threads of consecutive, increasing thread IDs\
    \ with the first warp containing thread 0."
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps inputs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'transformer_task': 'feature-extraction', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'last_hidden_state'}}, 'module_output_name': 'token_embeddings', 'architecture': 'BertModel'})
  (1): Pooling({'embedding_dimension': 384, 'pooling_mode': 'mean', 'include_prompt': True})
  (2): Normalize({'module_input_name': 'sentence_embedding', 'module_output_name': 'sentence_embedding'})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```
Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'How many ways does CUDA expose thread scopes to the programmer?',
    'CUDA/Software\nThread Block\nThread Block\nGrid\nThread Block\nThread Block\nThread Block\nShared Memory\nGlobal Memory\nCPU\nGPU 0\nGPU 1\n\n18\nThread Scopes\n• To account for non-uniform thread synchronization \ncosts, CUDA has introduced the notion of thread scopes. • A thread scope specifies which threads can \ncommunicate with each other using a primitive such as \nan atomic or a barrier. • Thread scopes are exposed to the programmer in 3 ways:\n• PTX\n• CUDA Math API\n• CUDA C++ \n• Always use the narrowest scope that ensures correctness \nof your application. • More on thread scopes in the GTC session [S62192]: \n“Advanced Performance Optimization in CUDA”.',
    '• Thread blocks in a cluster are guaranteed to be concurrently scheduled and enable efficient cooperation and data \nsharing for threads across multiple SMs. • For more information on this topic visit GTC session [S62192]: “Advanced Performance Optimization in CUDA”. Thread Block\nThread Block\nThread Block\nThread Block\nThread Block\nThread Block\nThread Block\nThread Block\nThread Block\nThread Block\nGrid\nThread Block\nThread Block\nThread Block\nThread Block\nThread Block\nThread Block\nThread Block Cluster\nThread Block Cluster\n\n12\nThread Hierarchy\nWhat about warps? • At runtime, a block of threads is divided into warps for SIMT execution. • The way a block is partitioned into warps is always the same. • Each warp contains threads of consecutive, increasing thread IDs with the first warp containing thread 0.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.7142, 0.4197],
#         [0.7142, 1.0000, 0.5994],
#         [0.4197, 0.5994, 1.0000]])
```
<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 149 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 100 samples:
  |          | sentence_0                                                                        | sentence_1                                                                            |
  |:---------|:----------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|
  | type     | string                                                                            | string                                                                                |
  | modality | text                                                                              | text                                                                                  |
  | details  | <ul><li>min: 8 tokens</li><li>mean: 18.94 tokens</li><li>max: 33 tokens</li></ul> | <ul><li>min: 115 tokens</li><li>mean: 202.22 tokens</li><li>max: 256 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                     | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
  |:---------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>What does NVIDIA's RTX Blackwell generative AI and neural rendering technology do for developers?</code> | <code>New Blackwell AI -based Neural Rendering and Neural Shading technologies will accelerate <br>developer usage of AI in their applications, including implementation and real<br>-time usage of <br>Generative AI-based rendering and simulation techniques. Generative AI will<br> help developers <br>dynamically create varied geometry, implement more realistic physical simulations, and enhance <br>iterative design processes. Professional 3D design applications can use RTX Blackwell Generative <br>AI capabilities to enable conversational workfl<br>ows to produce multiple design options faster than <br>ever, based on specified criteria, to more quickly iterate and fine<br>-tune parameters to create <br>optimal results. These and many other application scenarios will be supercharged by RTX <br>Blackwell generative AI and Neural Rendering capabilities. GPU performance and image quality has been continually improving, even with Moore’s Law <br>coming to an end, by using neural rendering techniques.</code> |
  | <code>What is YUV 4:2:0 chroma subsampling and why is it used instead of 4:4:4?</code>                         | <code>Chromasampling takes advantage of the fact that the human eye is more sensitive to changes in <br>luminance than chrominance. In YUV 4:4:4 video, each channel retains its full value in all channels; <br>however , this results in larger file sizes, and higher bandwidth required to transfer the video data. Chromasampling reduces the storage and bandwidth requirements by storing less information in <br>the video chrominance channels. For YUV 4:2:0 video, full information<br> is retained in the luminance <br>channel, but the two ch rominance channels contain only 25% of the original color information. This results in each video frame requiring half the data of an uncompressed 4:4:4 video frame, <br>with the tradeoff being a loss of color information. This color loss <br>does not imply low image <br>quality, standards from Blu -Ray through to HDR10 and streaming platforms today distribute <br>content to their audiences in a 4:2:0 format.</code>                                                       |
  | <code>How does Mega Geometry enable fast ray tracing of animated displaced subdivision surfaces?</code>        | <code>Subdivision surfaces are a type of geometry representation commonly used in film and other <br>production rendering workflows. Iterative refinement of a quad-based mesh with a subdivision <br>rule like Catmull-Clark, often with additional application of displacement maps, results in <br>smoothly rendered surfaces while maintaining high modeling efficiency and animation <br>friendliness. Fast ray tracing of subdivision surfaces is usually achieved by tessellating them into triangles. For <br>animations or changing viewpoints, new tessellations are needed each frame, leading to large <br>numbers of expensive BVH builds. Mega Geometry makes it possible for the application to map <br>its tessellation process directly to cluster generation, and construct BVHs from the resulting <br>CLASes extremely quickly. This method unlocks unprecedented real-time performance for ray <br>tracing of animated displaced subdivision surfaces.</code>                                                                    |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false,
      "directions": [
          "query_to_doc"
      ],
      "partition_mode": "joint",
      "hardness_mode": null,
      "hardness_strength": 0.0
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `num_train_epochs`: 4
- `per_device_eval_batch_size`: 16
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 16
- `num_train_epochs`: 4
- `max_steps`: -1
- `learning_rate`: 5e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: None
- `trackio_bucket_id`: None
- `trackio_static_space_id`: None
- `per_device_eval_batch_size`: 16
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `dataloader_multiprocessing_context`: None
- `dataloader_in_order`: True
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_static_graph`: None
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: None
- `fsdp_config`: None
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}
- `warmup_ratio`: None

</details>

### Training Time
- **Training**: 4.3 minutes

### Framework Versions
- Python: 3.14.2
- Sentence Transformers: 6.0.1
- Transformers: 5.16.1
- PyTorch: 2.14.0+cpu
- Accelerate: 1.14.0
- Datasets: 5.0.1
- Tokenizers: 0.23.2

## Additional Resources

- [Training and Finetuning Embedding Models with Sentence Transformers](https://huggingface.co/blog/train-sentence-transformers): the end-to-end guide for training or finetuning Sentence Transformer models.
- [Introduction to Matryoshka Embedding Models](https://huggingface.co/blog/matryoshka): variable-size embeddings that can be truncated with minimal quality loss.
- [Binary and Scalar Embedding Quantization for Significantly Faster & Cheaper Retrieval](https://huggingface.co/blog/embedding-quantization): post-training compression of embedding vectors.
- [Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/multimodal-sentence-transformers): use text, image, audio, and video models through the same API.
- [Training and Finetuning Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-multimodal-sentence-transformers): train multimodal embedding models, with a Visual Document Retrieval walkthrough.

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{oord2019representationlearningcontrastivepredictive,
      title={Representation Learning with Contrastive Predictive Coding},
      author={Aaron van den Oord and Yazhe Li and Oriol Vinyals},
      year={2019},
      eprint={1807.03748},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1807.03748},
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->