# K8s Assistant 部署包创建工具

本目录包含用于创建K8s Assistant原生部署包的脚本和工具。

## 📁 目录结构

```
deploy/
├── create-package.sh      # Linux/macOS 打包脚本
├── create-package.bat     # Windows 打包脚本
└── README.md             # 本说明文档
```

## 🚀 快速开始

### 创建部署包

#### Linux/macOS
```bash
cd deploy
chmod +x create-package.sh
./create-package.sh
```

#### Windows
```cmd
cd deploy
create-package.bat
```

## 📦 部署包内容

生成的部署包包含以下内容：

### 核心文件
- `backend/` - 后端服务代码
- `frontend/` - 前端服务代码
- `data_processing/` - 数据处理模块
- `docs/` - 项目文档

### 启动脚本
- `setup-native.sh` - 环境设置脚本
- `start-native.sh` - 服务启动脚本

### 配置文件
- `.env.template` - 环境配置模板
- `DEPLOY.md` - 部署说明文档
- `VERSION` - 版本信息

## 🔧 使用方法

### 创建部署包

1. **运行打包脚本**：
   ```bash
   # Linux/macOS
   ./create-package.sh
   
   # Windows
   create-package.bat
   ```

2. **获取压缩包**：
   脚本会在 `deploy/` 目录下生成一个时间戳命名的压缩包，例如：
   ```
   k8s-assistant-native-20241201-143022.tar.gz
   ```

3. **传输到目标机器**：
   将压缩包传输到需要部署的机器上。

### 在目标机器上部署

1. **解压部署包**：
   ```bash
   tar -xzf k8s-assistant-native-20241201-143022.tar.gz
   cd k8s-assistant-native-20241201-143022
   ```

2. **启动应用**：
   ```bash
   # 1. 设置环境
   ./setup-native.sh
   
   # 2. 配置API密钥
   cp .env.template .env
   # 编辑 .env 文件，设置 LLM_API_KEY
   
   # 3. 启动服务
   ./start-native.sh
   ```

## 📋 系统要求

### 必需依赖
- **Python 3.8+** - 后端服务运行环境
- **Node.js 16+** - 前端构建和运行环境
- **npm** - Node.js包管理器
- **curl** - 健康检查工具

### 推荐配置
- **内存**: 至少4GB RAM
- **存储**: 至少10GB可用空间
- **CPU**: 2核心以上

## 🔍 系统检查功能

系统检查脚本会验证以下项目：

### 操作系统
- 操作系统类型和版本
- 系统架构
- 内核版本

### Python环境
- Python版本（需要3.8+）
- pip包管理器
- venv模块
- 常用Python包

### Node.js环境
- Node.js版本（需要16+）
- npm包管理器
- npx工具

### 系统资源
- 内存大小
- 磁盘可用空间
- 端口占用情况

### 网络环境
- 互联网连接
- DNS解析
- 防火墙状态

### 开发工具
- git版本控制
- 编译工具
- 其他开发依赖

## 🛠️ 自定义配置

### 修改打包内容

编辑 `create-package.sh` 或 `create-package.bat`，可以：

1. **添加/移除文件**：
   修改文件复制部分，添加或移除需要打包的文件。

2. **修改目录结构**：
   调整目标目录结构，适应不同的部署需求。

3. **自定义脚本**：
   添加自定义的安装或配置脚本。

### 修改环境配置

编辑 `.env.example` 文件，可以：

1. **添加新的环境变量**
2. **修改默认值**
3. **添加配置说明**

## 🐛 故障排除

### 打包脚本问题

1. **权限问题**：
   ```bash
   chmod +x create-package.sh
   ```

2. **工具缺失**：
   确保安装了 `tar` 和 `gzip` 工具。

3. **路径问题**：
   确保在项目根目录的 `deploy/` 子目录中运行脚本。

### 系统检查问题

1. **检查失败**：
   查看具体的错误信息，安装缺失的依赖。

2. **警告信息**：
   警告不会阻止部署，但建议处理以获得最佳性能。

3. **网络问题**：
   确保网络连接正常，可以访问外部资源。

## 📚 相关文档

- [原生部署指南](../README-NATIVE.md) - 详细的部署说明
- [项目文档](../README.md) - 项目总体介绍
- [使用说明](../USAGE.md) - 应用使用指南

## 🤝 贡献

如需改进部署脚本或添加新功能，请：

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。
