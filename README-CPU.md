# K8s Assistant CPU 版本

## 概述

这是 K8s Assistant 的 CPU 版本，专门为无 GPU 环境优化，适合性能有限的机器。

## 主要优化

### 🚀 性能优化
- **移除 CUDA 依赖**：完全使用 CPU 版本的 PyTorch
- **减小镜像大小**：移除不必要的编译器和 CUDA 库
- **减少内存占用**：使用轻量级依赖包
- **加快启动速度**：减少下载和安装时间

### 📦 依赖优化
- **PyTorch CPU 版本**：`torch==2.1.1+cpu`
- **移除 spacy**：使用更轻量的文本处理库
- **最小化系统依赖**：只安装必要的系统包

## 快速开始

### 1. 使用 CPU 版本启动脚本

```bash
# 给脚本执行权限
chmod +x start-cpu.sh

# 启动 CPU 版本
./start-cpu.sh
```

### 2. 手动启动

```bash
# 使用 CPU 版本的 Dockerfile
docker-compose -f docker-compose.yml up -d
```

### 3. 开发环境

```bash
# 安装 CPU 版本的依赖
pip install -r backend/requirements-cpu.txt

# 运行测试
python backend/test_torch.py
```

## 文件说明

### CPU 版本专用文件
- `backend/requirements-cpu.txt` - CPU 版本的依赖文件
- `backend/Dockerfile.cpu` - CPU 版本的 Dockerfile
- `start-cpu.sh` - CPU 版本启动脚本
- `README-CPU.md` - 本说明文档

### 主要变化
- 移除了 `gcc` 和 `g++` 编译器
- 使用 `torch==2.1.1+cpu` 替代 GPU 版本
- 移除了 `spacy` 等重量级依赖
- 添加了 `CUDA_VISIBLE_DEVICES=""` 环境变量

## 性能对比

| 指标 | GPU 版本 | CPU 版本 |
|------|----------|----------|
| 镜像大小 | ~2.5GB | ~1.8GB |
| 启动时间 | 3-5分钟 | 2-3分钟 |
| 内存占用 | 4-6GB | 2-4GB |
| 推理速度 | 快 | 中等 |
| 适用环境 | 有 GPU | 无 GPU |

## 使用建议

### ✅ 推荐使用 CPU 版本的情况
- 开发环境
- 测试环境
- 无 GPU 的生产环境
- 资源有限的机器
- 快速原型验证

### ⚠️ 注意事项
- 文本嵌入和推理速度较慢
- 不适合大规模并发处理
- 建议用于小到中等规模的应用

## 故障排除

### 常见问题

1. **网络连接问题 (apt-get 失败)**
   ```bash
   # 测试镜像源连接性
   chmod +x test-sources.sh
   ./test-sources.sh
   
   # 使用网络优化启动脚本
   chmod +x start-cpu-network.sh
   ./start-cpu-network.sh
   
   # 如果还是失败，使用备用 Dockerfile
   docker-compose -f docker-compose.yml build --no-cache backend
   # 或者手动指定备用 Dockerfile
   docker build -f backend/Dockerfile.cpu.backup -t k8s-assistant-backend ./backend
   ```

2. **安装速度慢**
   ```bash
   # 使用国内镜像源
   pip install -r requirements-cpu.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
   ```

3. **内存不足**
   ```bash
   # 减少并发数
   export MAX_WORKERS=1
   ```

4. **启动失败**
   ```bash
   # 查看详细日志
   docker-compose logs backend
   
   # 清理缓存后重试
   docker system prune -f
   docker-compose build --no-cache
   ```

5. **代理问题**
   ```bash
   # 如果使用代理，设置 Docker 代理
   export HTTP_PROXY=http://your-proxy:port
   export HTTPS_PROXY=http://your-proxy:port
   ```

## 技术支持

如果遇到问题，请：
1. 查看日志文件
2. 运行测试脚本：`python backend/test_torch.py`
3. 检查环境变量配置
4. 确认系统资源是否充足

---

**CPU 版本专为无 GPU 环境设计，提供更好的兼容性和更低的资源占用！**
