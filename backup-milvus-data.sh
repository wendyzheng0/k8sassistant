#!/bin/bash

# K8s Assistant - Milvus 数据备份和恢复脚本

BACKUP_DIR="./milvus_backups"
DATA_DIR="./milvus_embedded_data"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "💾 Milvus 数据备份和恢复工具"
echo "=================================="

# 创建备份目录
mkdir -p "$BACKUP_DIR"

backup_data() {
    echo "📤 开始备份 Milvus 数据..."
    
    if [ ! -d "$DATA_DIR" ]; then
        echo "❌ 数据目录不存在: $DATA_DIR"
        exit 1
    fi
    
    if [ -z "$(ls -A $DATA_DIR 2>/dev/null)" ]; then
        echo "⚠️  数据目录为空，跳过备份"
        return
    fi
    
    BACKUP_FILE="$BACKUP_DIR/milvus_backup_$TIMESTAMP.tar.gz"
    
    echo "📁 备份目录: $DATA_DIR"
    echo "💾 备份文件: $BACKUP_FILE"
    
    tar -czf "$BACKUP_FILE" -C "$(dirname "$DATA_DIR")" "$(basename "$DATA_DIR")"
    
    if [ $? -eq 0 ]; then
        echo "✅ 备份完成: $BACKUP_FILE"
        echo "📊 备份大小: $(du -h "$BACKUP_FILE" | cut -f1)"
    else
        echo "❌ 备份失败"
        exit 1
    fi
}

restore_data() {
    echo "📥 开始恢复 Milvus 数据..."
    
    if [ -z "$1" ]; then
        echo "❌ 请指定要恢复的备份文件"
        echo "用法: $0 restore <backup_file>"
        exit 1
    fi
    
    BACKUP_FILE="$1"
    
    if [ ! -f "$BACKUP_FILE" ]; then
        echo "❌ 备份文件不存在: $BACKUP_FILE"
        exit 1
    fi
    
    echo "⚠️  警告：这将覆盖现有的 Milvus 数据！"
    read -p "确认恢复？(y/N): " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ 取消恢复操作"
        exit 0
    fi
    
    echo "🔄 停止服务..."
    docker-compose down 2>/dev/null || true
    
    echo "🧹 清理现有数据..."
    if [ -d "$DATA_DIR" ]; then
        rm -rf "$DATA_DIR"/*
    fi
    
    echo "📥 恢复数据..."
    tar -xzf "$BACKUP_FILE" -C "$(dirname "$DATA_DIR")"
    
    if [ $? -eq 0 ]; then
        echo "✅ 数据恢复完成"
        echo "📁 恢复的目录: $DATA_DIR"
    else
        echo "❌ 数据恢复失败"
        exit 1
    fi
}

list_backups() {
    echo "📋 可用的备份文件:"
    echo "===================="
    
    if [ ! -d "$BACKUP_DIR" ] || [ -z "$(ls -A $BACKUP_DIR 2>/dev/null)" ]; then
        echo "📭 没有找到备份文件"
        return
    fi
    
    for backup in "$BACKUP_DIR"/milvus_backup_*.tar.gz; do
        if [ -f "$backup" ]; then
            size=$(du -h "$backup" | cut -f1)
            date_str=$(echo "$(basename "$backup")" | sed 's/milvus_backup_\([0-9]\{8\}\)_\([0-9]\{6\}\)\.tar\.gz/\1 \2/' | sed 's/\([0-9]\{4\}\)\([0-9]\{2\}\)\([0-9]\{2\}\)_\([0-9]\{2\}\)\([0-9]\{2\}\)\([0-9]\{2\}\)/\1-\2-\3 \4:\5:\6/')
            echo "📦 $(basename "$backup")"
            echo "   📅 时间: $date_str"
            echo "   💾 大小: $size"
            echo ""
        fi
    done
}

cleanup_backups() {
    echo "🧹 清理旧备份文件..."
    
    if [ ! -d "$BACKUP_DIR" ]; then
        echo "📭 备份目录不存在"
        return
    fi
    
    echo "⚠️  警告：这将删除所有备份文件！"
    read -p "确认删除？(y/N): " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ 取消清理操作"
        return
    fi
    
    rm -rf "$BACKUP_DIR"/*
    echo "✅ 备份文件清理完成"
}

show_help() {
    echo "用法: $0 [命令] [参数]"
    echo ""
    echo "命令:"
    echo "  backup          备份 Milvus 数据"
    echo "  restore <file>  从备份文件恢复数据"
    echo "  list            列出可用的备份文件"
    echo "  cleanup         清理所有备份文件"
    echo "  help            显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 backup                    # 备份数据"
    echo "  $0 restore backup_file.tar.gz # 恢复数据"
    echo "  $0 list                      # 列出备份"
    echo "  $0 cleanup                   # 清理备份"
}

# 主逻辑
case "${1:-help}" in
    "backup")
        backup_data
        ;;
    "restore")
        restore_data "$2"
        ;;
    "list")
        list_backups
        ;;
    "cleanup")
        cleanup_backups
        ;;
    "help"|*)
        show_help
        ;;
esac
