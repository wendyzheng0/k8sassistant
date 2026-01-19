#!/usr/bin/env python3
"""
CLI Entry Point
统一命令行入口
"""

import os
import sys
import argparse
import asyncio
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="K8s Assistant 文档处理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用 Milvus 存储
  python -m data_processing.processors.cli --data-dir ./data/zh-cn --backend milvus

  # 使用 Elasticsearch 存储
  python -m data_processing.processors.cli --data-dir ./data/zh-cn --backend elasticsearch

  # 自定义分块大小
  python -m data_processing.processors.cli --data-dir ./data/zh-cn --chunk-size 512 --chunk-overlap 50
        """
    )
    
    # 数据源配置
    parser.add_argument(
        "--data-dir", "-d",
        type=str,
        default=None,
        help="文档数据目录路径 (默认: ./data/zh-cn)"
    )
    
    # 存储后端配置
    parser.add_argument(
        "--backend", "-b",
        type=str,
        choices=["milvus", "elasticsearch"],
        default="milvus",
        help="存储后端类型 (默认: milvus)"
    )
    
    # Milvus 配置
    parser.add_argument(
        "--milvus-uri",
        type=str,
        default=None,
        help="Milvus 服务地址 (默认: http://localhost:19530)"
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default=None,
        help="Milvus 集合名称 (默认: k8s_docs)"
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="不覆盖已存在的集合"
    )
    
    # Elasticsearch 配置
    parser.add_argument(
        "--es-host",
        type=str,
        default=None,
        help="Elasticsearch 服务地址 (默认: http://localhost:9200)"
    )
    parser.add_argument(
        "--es-index",
        type=str,
        default=None,
        help="Elasticsearch 索引名称 (默认: k8s-docs)"
    )
    
    # 文本处理配置
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="文本块大小 (默认: 1024)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="文本块重叠大小 (默认: 100)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="批处理大小 (默认: 32)"
    )
    
    # 其他配置
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="日志级别 (默认: INFO)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="试运行模式，只读取和处理文档，不存储"
    )
    
    return parser


async def main_async(args: argparse.Namespace) -> int:
    """异步主函数"""
    import logging
    
    # 配置日志
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    from .runner import PipelineRunner
    from .config import ProcessorSettings
    
    # 构建配置参数
    config_kwargs = {}
    
    if args.data_dir:
        config_kwargs["data_dir"] = args.data_dir
    if args.backend:
        config_kwargs["storage_backend"] = args.backend
    if args.milvus_uri:
        config_kwargs["milvus_uri"] = args.milvus_uri
    if args.collection_name:
        config_kwargs["collection_name"] = args.collection_name
    if args.no_overwrite:
        config_kwargs["milvus_overwrite"] = False
    if args.es_host:
        config_kwargs["es_host"] = args.es_host
    if args.es_index:
        config_kwargs["es_index"] = args.es_index
    if args.chunk_size:
        config_kwargs["chunk_size"] = args.chunk_size
    if args.chunk_overlap:
        config_kwargs["chunk_overlap"] = args.chunk_overlap
    if args.batch_size:
        config_kwargs["batch_size"] = args.batch_size
    if args.log_level:
        config_kwargs["log_level"] = args.log_level
    
    try:
        # 创建配置
        settings = ProcessorSettings(**config_kwargs)
        
        # 验证数据目录
        if not os.path.exists(settings.data_dir):
            print(f"❌ Error: Data directory does not exist: {settings.data_dir}")
            return 1
        
        print("=" * 60)
        print("🚀 K8s Assistant Document Processor")
        print("=" * 60)
        
        if args.dry_run:
            print("⚠️  DRY RUN MODE - Documents will not be stored")
        
        # 运行流水线
        runner = PipelineRunner(settings)
        result = await runner.run()
        
        # 打印结果
        print("\n" + "=" * 60)
        print("📊 Processing Result")
        print("=" * 60)
        print(result)
        
        if result.errors:
            print("\n⚠️  Errors encountered:")
            for i, error in enumerate(result.errors[:10], 1):
                print(f"   {i}. {error}")
            if len(result.errors) > 10:
                print(f"   ... and {len(result.errors) - 10} more errors")
        
        return 0 if result.success else 1
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main() -> int:
    """主函数入口"""
    parser = create_parser()
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())

