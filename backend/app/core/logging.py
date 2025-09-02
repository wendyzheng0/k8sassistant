"""
日志配置管理
"""

import sys
import os
from pathlib import Path
from loguru import logger
from app.core.config import settings


def setup_logging():
    """配置日志系统"""
    
    # 移除默认的日志处理器
    logger.remove()
    
    # 创建日志目录
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # 控制台日志格式
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    
    # 文件日志格式
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss} | "
        "{level: <8} | "
        "{name}:{function}:{line} | "
        "{message}"
    )
    
    # 添加控制台处理器
    logger.add(
        sys.stdout,
        format=console_format,
        level=settings.LOG_LEVEL,
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # 添加文件处理器 - 所有日志
    logger.add(
        log_dir / "app.log",
        format=file_format,
        level="DEBUG",
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        backtrace=True,
        diagnose=True
    )
    
    # 添加文件处理器 - 错误日志
    logger.add(
        log_dir / "error.log",
        format=file_format,
        level="ERROR",
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        backtrace=True,
        diagnose=True
    )
    
    # 设置日志级别
    logger.level(settings.LOG_LEVEL)
    
    return logger


def get_logger(name: str = None):
    """获取日志记录器"""
    if name:
        return logger.bind(name=name)
    return logger
