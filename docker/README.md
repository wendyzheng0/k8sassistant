# Docker 部署目录

这个目录包含了所有与 Docker 部署相关的文件和配置。

## 目录结构

```
docker/
├── docker-compose.yml    # Docker Compose 配置文件
├── start.sh             # Linux/macOS Docker 启动脚本
├── start-cpu.sh         # Linux/macOS Docker CPU 版本启动脚本
├── nginx/               # Nginx 配置目录
│   ├── nginx.conf       # Nginx 配置文件
│   └── ssl/             # SSL 证书目录
└── README.md            # 本文件
```

## 使用方法

### 从根目录启动（推荐）

在项目根目录下运行：

**Linux/macOS:**
```bash
# 标准版本
./start-docker.sh

# CPU 版本
./start-docker-cpu.sh
```

**Windows:**
```cmd
# 标准版本
start-docker.bat

# CPU 版本
start-docker-cpu.bat
```

### 从 docker 目录启动

进入 docker 目录后运行：

```bash
cd docker

# 启动服务
docker-compose up -d

# 停止服务
docker-compose down

# 查看日志
docker-compose logs -f
```

## 服务说明

- **frontend**: 前端应用，端口 3000
- **backend**: 后端 API 服务，端口 8000
- **nginx**: 反向代理，端口 80/443

## 环境配置

启动前请确保：

1. 已安装 Docker 和 Docker Compose
2. 已配置 `.env` 文件（从 `.env.example` 复制）
3. 已配置必要的环境变量，特别是 `LLM_API_KEY`

## 故障排除

如果遇到问题，可以：

1. 查看服务状态：`docker-compose ps`
2. 查看日志：`docker-compose logs -f [service_name]`
3. 重新构建镜像：`docker-compose build --no-cache`
4. 清理系统：`docker system prune -f`
