from milvus_lite.server import Server
from pymilvus import connections
import time
import traceback
import signal
import sys
import threading

milvus_server = None

def start_milvus():
    global milvus_server
    MILVUS_DATA='milvus_data'
    MILVUS_PORT=19530
    
    try:
        print("🚀 Starting Milvus server...")
        milvus_server = Server(db_file=MILVUS_DATA, address='localhost:19530')
        milvus_server.start()
        print("✅ Milvus server started successfully")
        
        # 等待服务器完全启动
        print("⏳ Waiting for server to fully start...")
        time.sleep(3)
        
        # 测试连接
        try:
            connections.connect("default", host="localhost", port="19530")
            print("✅ Milvus connection test successful")
            connections.disconnect("default")
        except Exception as e:
            print(f"⚠️ Milvus connection test failed: {e}")
            
    except Exception as e:
        print(f"🔴 Milvus server startup failed: {e}")
        milvus_server = None
        raise


def signal_handler(signum, frame):
    """信号处理器，用于优雅地关闭服务器"""
    print(f"\n🛑 Received signal {signum}, shutting down Milvus server...")
    if milvus_server is not None:
        try:
            milvus_server.stop()
            print("✅ Milvus server stopped successfully")
        except Exception as e:
            print(f"⚠️ Error occurred while stopping Milvus server: {e}")
    print("👋 Program exiting")
    sys.exit(0)


def keep_alive():
    """保持进程运行的函数"""
    print("🔄 Milvus server is running...")
    print("💡 Press Ctrl+C or send SIGTERM signal to stop the server")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Received keyboard interrupt signal")
        signal_handler(signal.SIGINT, None)


if __name__ == "__main__":
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # 终止信号
    
    try:
        # 启动 Milvus 服务器
        start_milvus()
        
        # 保持进程运行
        keep_alive()
        
    except Exception as e:
        print(f"🔴 Program execution error: {e}")
        traceback.print_exc()
        # 确保在异常情况下也停止服务器
        if milvus_server is not None:
            try:
                milvus_server.stop()
                print("✅ Milvus server stopped")
            except:
                pass
        sys.exit(1)