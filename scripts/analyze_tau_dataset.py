#!/usr/bin/env python3
"""
TAU Urban Acoustic Scenes 2022 数据集分析脚本
用于T10: 数据集验收与重组
"""
import pandas as pd
import json
import os
from pathlib import Path
import librosa
import soundfile as sf
import numpy as np
from collections import Counter

def main():
    # 路径配置
    meta_path = 'data/TAU-urban-acoustic-scenes-2022-mobile-development/development/meta.csv'
    audio_dir = 'data/TAU-urban-acoustic-scenes-2022-mobile-development/development/'
    output_json = 'artifacts/tau_scenes.json'
    
    print("=== TAU Urban Acoustic Scenes 2022 数据集分析 ===\n")
    
    # 1. 读取meta.csv
    print("[1/4] 读取meta.csv...")
    meta = pd.read_csv(meta_path, sep='\t')
    print(f"✅ 总音频数: {len(meta)}")
    print(f"列名: {list(meta.columns)}")
    
    # 2. 场景分布统计
    print("\n[2/4] 场景分布统计...")
    scene_counts = meta['scene_label'].value_counts()
    print("\n场景类型及数量:")
    for scene, count in scene_counts.items():
        print(f"  - {scene}: {count}个")
    
    # 3. 抽样验证音频文件
    print("\n[3/4] 抽样验证音频文件...")
    sample_size = min(10, len(meta))
    sample_files = meta['filename'].sample(n=sample_size, random_state=1337)
    
    valid_count = 0
    sr_list = []
    duration_list = []
    
    for filename in sample_files:
        filepath = os.path.join(audio_dir, filename)
        try:
            # 读取音频信息
            info = sf.info(filepath)
            sr_list.append(info.samplerate)
            duration_list.append(info.duration)
            valid_count += 1
        except Exception as e:
            print(f"  ❌ 文件读取失败: {filename} - {e}")
    
    print(f"✅ 验证 {valid_count}/{sample_size} 个文件成功")
    
    if sr_list:
        print(f"  采样率: {Counter(sr_list).most_common(1)[0][0]} Hz (最常见)")
        print(f"  平均时长: {np.mean(duration_list):.2f} 秒")
    
    # 4. 生成JSON输出
    print("\n[4/4] 生成输出JSON...")
    output = {
        "dataset": "TAU Urban Acoustic Scenes 2022 Mobile - Development",
        "total_files": int(len(meta)),
        "scenes": {
            scene: int(count) 
            for scene, count in scene_counts.items()
        },
        "sample_rate": int(Counter(sr_list).most_common(1)[0][0]) if sr_list else None,
        "average_duration_sec": float(np.mean(duration_list)) if duration_list else None,
        "verified_samples": valid_count,
        "devices": list(meta['source_label'].unique()) if 'source_label' in meta.columns else []
    }
    
    os.makedirs('artifacts', exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"✅ 输出已保存到: {output_json}")
    print(f"\n场景列表: {list(output['scenes'].keys())}")
    
    # 5. 验证LibriTTS数据
    print("\n" + "="*50)
    print("LibriTTS 数据验证")
    print("="*50)
    
    libritts_dir = 'data/raw/speech/libritts/5339'
    if os.path.exists(libritts_dir):
        wav_files = list(Path(libritts_dir).glob('*.wav'))
        print(f"✅ LibriTTS speaker 5339: {len(wav_files)} 个文件")
        
        # 抽样验证
        if wav_files:
            sample_wav = wav_files[0]
            try:
                info = sf.info(sample_wav)
                print(f"  采样率: {info.samplerate} Hz")
                print(f"  时长示例: {info.duration:.2f} 秒")
            except Exception as e:
                print(f"  ⚠️  音频读取警告: {e}")
    else:
        print(f"⚠️  LibriTTS路径不存在: {libritts_dir}")
    
    print("\n✅ T10数据验收完成！")

if __name__ == '__main__':
    main()

