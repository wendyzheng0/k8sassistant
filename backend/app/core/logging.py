"""
日志配置管理
"""

import sys
import os
import logging
from pathlib import Path
from loguru import logger
from app.core.config import settings


class InterceptHandler(logging.Handler):
    """
    拦截标准库 logging 的日志，重定向到 loguru
    这样 shared 模块使用标准库 logging 写的日志也能被 loguru 捕获并输出
    """
    def emit(self, record: logging.LogRecord) -> None:
        # 获取对应的 loguru level
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # 找到调用者的帧
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


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
    
    # 拦截标准库 logging，重定向到 loguru
    # 这样 shared 模块等使用标准库 logging 的代码也能正常输出日志
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    return logger


def get_logger(name: str = None):
    """获取日志记录器"""
    if name:
        return logger.bind(name=name)
    return logger
