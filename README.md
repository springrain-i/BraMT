# Srtp_Hybrid

## Commit 格式规范
Commit 分为“标题”和“内容”。原则上标题全部小写。内容首字母大写。

### 标题
`<type>](<scope>) <subject> (#pr)`
  
`<type>`:
本次提交的类型，暂时限定在以下类型

- feat: 新功能（feature）
- fix: 修补bug
- improvement：原有功能的优化和改进
- revert：回滚
- docs: 文档（documentation）
- performance/optimize：性能优化

几点说明：
+ 如在一次提交中出现多种类型，需增加多个类型。

`<scope>`:
本次提交的影响范围，例子如下
- hybrid_model：混合模型
- finetune_main: 微调主程序
- logger: 日志

`<subject>`: 标题需尽量清晰表明本次提交的主要内容

### 内容


最终示例
```
[fix][improvement](log)(lr)    修改log记录方式，增添细致学习率控制

1. 删去log中记录model的冗杂功能
2. trainer中增添attn，mamba分层学习率控制
```