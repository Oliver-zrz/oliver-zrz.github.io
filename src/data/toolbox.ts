// 网站文案独立维护，Skill 原始描述保留在自动导出的 skills.json 中。
export const toolboxGroups = [
  {
    id: 'writing', title: '写作与记录', description: '把工作过程收好，再整理成文章。',
    tools: [
      {name: 'work-journal', title: '工作记录', summary: '按任务保存过程、结论和验证结果，再从记录中整理博客。'},
      {name: 'shuorenhua', title: '文字整理', summary: '改顺句子，删掉套话，保留原意和技术细节。'},
      {name: 'diagram-design', title: '画图', summary: '画架构、流程、时序和数据流，保存为 SVG 或可编辑的 Excalidraw 文件。'},
    ],
  },
  {
    id: 'environment', title: '昇腾环境与迁移', description: '先看环境，再动代码。',
    tools: [
      {name: 'ascendc-env-check', title: 'Ascend C 环境检查', summary: '查看 NPU 状态、CANN 配置与开发依赖，确认目标芯片架构。'},
      {name: 'hccl-env-check', title: 'HCCL 环境检查', summary: '只读检查 HCCL/HCOMM 仓库、CANN 路径和环境变量，列出缺项。'},
      {name: 'npu-smi', title: 'NPU 设备管理', summary: '查设备健康、温度、内存和进程，也可查找配置、固件与虚拟化命令。'},
      {name: 'npu-arch', title: 'NPU 架构速查', summary: '从芯片型号查架构代际、特性支持和条件编译差异。'},
      {name: 'migration-ascend-torchnpu-skills', title: '模型迁移', summary: '把 CPU/GPU 上的 PyTorch 模型适配到昇腾 NPU，并检查迁移结果。'},
      {name: 'migration-ascend-torchnpu-skills-environment-setup', title: '迁移环境准备', summary: '准备驱动、CANN 和 torch_npu，检查版本是否匹配。'},
      {name: 'migration-ascend-torchnpu-skills-migration-execution', title: '迁移代码适配', summary: '替换设备与接口，处理迁移中遇到的不兼容问题。'},
      {name: 'migration-ascend-torchnpu-skills-torch-npu-reference', title: 'torch_npu 接口速查', summary: '查接口支持情况，以及 CPU/GPU 接口在 NPU 上的对应写法。'},
    ],
  },
  {
    id: 'operators', title: '算子与测试', description: '查错误、看精度，也把用例补上。',
    tools: [
      {name: 'ascendc-operator-precision-debug', title: '算子精度排查', summary: '从误差分布和代码检查入手，定位 allclose 失败、NaN 或全零等问题。'},
      {name: 'ascendc-runtime-debug', title: '算子运行错误', summary: '结合 aclnn 错误码和 plog，排查 Tiling、Kernel 查找及环境问题。'},
      {name: 'ops-precision-standard', title: '精度标准', summary: '查 FP16、FP32、BF16 等数据类型的 atol/rtol 验收要求。'},
      {name: 'gen-torch-npu-api-ut', title: 'API 功能用例', summary: '按 torch_npu 项目的 unittest 与 TestCase 风格生成 API 测试。'},
      {name: 'community-pr-test-tracking', title: '社区 PR 测试', summary: '从 PyTorch PR 提取用例，适配到 NPU 后执行并记录结果；跳过单独标明。'},
    ],
  },
  {
    id: 'performance', title: '性能与图模式', description: '从 profiling 数据找到耗时的地方。',
    tools: [
      {name: 'profiling-analysis', title: 'Profiling 分析', summary: '读取采集文件，区分计算、通信与下发耗时，逐步缩小排查范围。'},
      {name: 'profiling-analysis-communication', title: '通信分析', summary: '检查通信效率、数据传输和 rank 间的快慢差异。'},
      {name: 'profiling-analysis-computing', title: '计算分析', summary: '筛选高耗时算子，结合指令占比、shape 和 Roofline 判断计算瓶颈。'},
      {name: 'profiling-analysis-hostbound', title: 'Host 与快慢卡分析', summary: '结合 Insight 和系统 trace，查看卡间差异及 Host 进程耗时。'},
      {name: 'profiling-analysis-roofline', title: 'Roofline 核对', summary: '对 aic_mac_ratio ≥ 80% 的算子，比较理论耗时与实际耗时。'},
      {name: 'torch-npugraph-ex-knowledge', title: 'npugraph_ex 使用参考', summary: '查 Capture & Replay、编译缓存、多流、内存复用和调试方法。'},
      {name: 'torch-npugraph-ex-template', title: 'npugraph_ex 最小示例', summary: '提供基础编译与 cache_compile 模板，用来开始实验或复现问题。'},
    ],
  },
];
