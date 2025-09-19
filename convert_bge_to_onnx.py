#!/usr/bin/env python3
"""
专门转换BGE模型为ONNX格式
"""

import os
import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import argparse
from pathlib import Path

def get_expected_dimension(model_name: str) -> int:
    """
    根据模型名称获取期望的嵌入维度
    """
    model_name_lower = model_name.lower()
    
    # 检查是否是特定的512维模型版本
    if "7999e1d3359715c523056ef9478215996d62a620" in model_name:
        return 512  # 这个特定的BGE模型版本确实是512维
    elif "bge-small" in model_name_lower:
        return 384  # 大多数BGE-small模型是384维
    elif "bge-base" in model_name_lower:
        return 768
    elif "bge-large" in model_name_lower:
        return 1024
    elif "all-minilm-l6-v2" in model_name_lower:
        return 384
    else:
        # 默认返回None，让系统自动检测
        return None

def convert_bge_to_onnx(
    model_name: str = "BAAI/bge-small-zh-v1.5",
    output_dir: str = "bge_onnx_model",
    max_length: int = 512,
    expected_dimension: int = None
):
    """
    转换BGE模型为ONNX格式
    """
    
    print(f"🔄 转换BGE模型: {model_name}")
    print(f"📁 输出目录: {output_dir}")
    if expected_dimension:
        print(f"📏 期望维度: {expected_dimension}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 检查是否是本地路径
        if os.path.exists(model_name):
            print(f"📁 使用本地模型路径: {model_name}")
            local_path = model_name
        else:
            print(f"🌐 使用远程模型: {model_name}")
            local_path = model_name
        
        # 加载tokenizer和模型
        print("📥 加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(local_path)
        
        print("📥 加载模型...")
        model = AutoModel.from_pretrained(local_path)
        model.eval()
        model = model.cpu()  # 强制CPU
        
        # 检测模型维度
        print("📏 检测模型维度...")
        model_dimension = model.config.hidden_size
        print(f"✅ 模型隐藏层维度: {model_dimension}")
        
        # 验证维度
        if expected_dimension and model_dimension != expected_dimension:
            print(f"⚠️ 维度不匹配: 期望 {expected_dimension}, 实际 {model_dimension}")
            print("💡 继续转换，但请注意维度差异")
        elif expected_dimension:
            print(f"✅ 维度匹配: {model_dimension}")
        
        # 创建示例输入
        print("🔧 创建示例输入...")
        sample_text = "这是一个测试文本。"
        
        inputs = tokenizer(
            sample_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
        
        # 确保所有输入都在CPU上
        inputs = {k: v.cpu() for k, v in inputs.items()}
        
        # BGE模型通常支持token_type_ids
        has_token_type_ids = "token_type_ids" in inputs
        print(f"✅ 模型支持token_type_ids: {has_token_type_ids}")
        
        if has_token_type_ids:
            input_names = ["input_ids", "attention_mask", "token_type_ids"]
            onnx_inputs = (
                inputs["input_ids"],
                inputs["attention_mask"], 
                inputs["token_type_ids"]
            )
            dynamic_axes = {
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "token_type_ids": {0: "batch", 1: "sequence"}
            }
        else:
            input_names = ["input_ids", "attention_mask"]
            onnx_inputs = (
                inputs["input_ids"],
                inputs["attention_mask"]
            )
            dynamic_axes = {
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"}
            }
        
        # 测试模型前向传播
        print("🧪 测试模型前向传播...")
        with torch.no_grad():
            if has_token_type_ids:
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    token_type_ids=inputs["token_type_ids"]
                )
            else:
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"]
                )
        
        print(f"✅ 模型输出形状: {outputs.last_hidden_state.shape}")
        
        # 验证输出维度
        actual_dimension = outputs.last_hidden_state.shape[-1]
        print(f"📏 实际输出维度: {actual_dimension}")
        
        if expected_dimension and actual_dimension != expected_dimension:
            print(f"❌ 输出维度不匹配: 期望 {expected_dimension}, 实际 {actual_dimension}")
            return False
        elif expected_dimension:
            print(f"✅ 输出维度匹配: {actual_dimension}")
        
        # 导出ONNX模型
        output_path = os.path.join(output_dir, "model.onnx")
        print(f"📤 导出ONNX模型到: {output_path}")
        
        # 删除旧的模型文件（如果存在）
        if os.path.exists(output_path):
            os.remove(output_path)
            print("🗑️ 删除旧的ONNX模型文件")
        
        torch.onnx.export(
            model,
            onnx_inputs,
            output_path,
            input_names=input_names,
            output_names=["last_hidden_state", "pooler_output"],
            dynamic_axes=dynamic_axes,
            opset_version=14,
            do_constant_folding=True,
            export_params=True,
            verbose=False
        )
        
        # 验证导出的模型
        print("🔍 验证导出的ONNX模型...")
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
            print("✅ ONNX模型加载成功")
            
            # 检查输入输出
            print("📋 模型输入:")
            for input_meta in session.get_inputs():
                print(f"  - {input_meta.name}: {input_meta.shape} ({input_meta.type})")
            
            print("📋 模型输出:")
            for output_meta in session.get_outputs():
                print(f"  - {output_meta.name}: {output_meta.shape} ({output_meta.type})")
            
            # 测试推理
            print("🧪 测试ONNX模型推理...")
            if has_token_type_ids:
                onnx_inputs_dict = {
                    "input_ids": inputs["input_ids"].numpy(),
                    "attention_mask": inputs["attention_mask"].numpy(),
                    "token_type_ids": inputs["token_type_ids"].numpy()
                }
            else:
                onnx_inputs_dict = {
                    "input_ids": inputs["input_ids"].numpy(),
                    "attention_mask": inputs["attention_mask"].numpy()
                }
            
            onnx_outputs = session.run(None, onnx_inputs_dict)
            print(f"✅ ONNX推理成功! 输出形状: {onnx_outputs[0].shape}")
            
            # 验证ONNX模型输出维度
            onnx_dimension = onnx_outputs[0].shape[-1]
            print(f"📏 ONNX模型输出维度: {onnx_dimension}")
            
            if expected_dimension and onnx_dimension != expected_dimension:
                print(f"❌ ONNX模型维度不匹配: 期望 {expected_dimension}, 实际 {onnx_dimension}")
                return False
            elif expected_dimension:
                print(f"✅ ONNX模型维度匹配: {onnx_dimension}")
            
        except Exception as e:
            print(f"❌ ONNX模型验证失败: {e}")
            return False
        
        # 保存tokenizer配置
        print("💾 保存tokenizer配置...")
        tokenizer.save_pretrained(output_dir)
        
        # 保存模型配置
        print("💾 保存模型配置...")
        model.config.save_pretrained(output_dir)
        
        # 保存维度信息
        print("💾 保存维度信息...")
        dimension_info = {
            "model_name": model_name,
            "hidden_size": model_dimension,
            "output_dimension": actual_dimension,
            "expected_dimension": expected_dimension,
            "max_length": max_length,
            "conversion_timestamp": str(np.datetime64('now'))
        }
        
        import json
        dimension_file = os.path.join(output_dir, "dimension_info.json")
        with open(dimension_file, 'w', encoding='utf-8') as f:
            json.dump(dimension_info, f, indent=2, ensure_ascii=False)
        
        print("🎉 BGE模型转换完成!")
        print(f"📁 输出目录: {output_dir}")
        print(f"📄 ONNX模型: {output_path}")
        print(f"📄 维度信息: {dimension_file}")
        print(f"📏 最终维度: {actual_dimension}")
        
        return True
        
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='转换BGE模型为ONNX格式')
    parser.add_argument('--model-name', type=str, default='BAAI/bge-small-zh-v1.5',
                        help='BGE模型名称')
    parser.add_argument('--output-dir', type=str, default='bge_onnx_model',
                        help='输出目录')
    parser.add_argument('--max-length', type=int, default=512,
                        help='最大序列长度')
    parser.add_argument('--expected-dimension', type=int, default=None,
                        help='期望的嵌入维度 (例如: 384 for bge-small, 768 for bge-base)')
    
    args = parser.parse_args()
    
    # 如果没有指定期望维度，尝试自动检测
    expected_dimension = args.expected_dimension
    if expected_dimension is None:
        expected_dimension = get_expected_dimension(args.model_name)
        if expected_dimension:
            print(f"🔍 自动检测到期望维度: {expected_dimension}")
    
    success = convert_bge_to_onnx(
        model_name=args.model_name,
        output_dir=args.output_dir,
        max_length=args.max_length,
        expected_dimension=expected_dimension
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
