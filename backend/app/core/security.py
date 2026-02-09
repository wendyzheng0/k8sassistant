"""
安全工具函数
提供密码哈希、JWT认证等安全相关功能
"""

from datetime import datetime, timedelta
from typing import Optional
import uuid
import jwt
from passlib.context import CryptContext
from loguru import logger

from app.core.config import settings

# 密码哈希上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# 匿名用户 ID 前缀
ANONYMOUS_ID_PREFIX = "anon_"
# 匿名 ID 长度（不包含前缀）
ANONYMOUS_ID_LENGTH = 32


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码"""
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        logger.error(f"❌ Password verification failed: {e}")
        return False


def get_password_hash(password: str) -> str:
    """获取密码哈希"""
    try:
        return pwd_context.hash(password)
    except Exception as e:
        logger.error(f"❌ Password hashing failed: {e}")
        raise


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """创建访问令牌"""
    try:
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm="HS256")
        
        logger.debug(f"✅ Created access token for user: {data.get('sub', 'unknown')}")
        return encoded_jwt
        
    except Exception as e:
        logger.error(f"❌ Failed to create access token: {e}")
        raise


def verify_token(token: str) -> Optional[dict]:
    """验证令牌"""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        user_id: str = payload.get("sub")
        if user_id is None:
            return None
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("⚠️ Token has expired")
        return None
    except jwt.PyJWTError as e:
        logger.error(f"❌ Token verification failed: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ Unexpected error during token verification: {e}")
        return None


def create_anonymous_token(anonymous_id: str) -> str:
    """创建匿名用户令牌"""
    try:
        expire = datetime.utcnow() + timedelta(days=30)  # 匿名用户令牌有效期30天
        to_encode = {
            "sub": anonymous_id,
            "type": "anonymous",
            "exp": expire
        }
        encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm="HS256")
        
        logger.debug(f"✅ Created anonymous token for ID: {anonymous_id}")
        return encoded_jwt
        
    except Exception as e:
        logger.error(f"❌ Failed to create anonymous token: {e}")
        raise


def extract_user_id_from_token(token: str) -> Optional[str]:
    """从令牌中提取用户ID"""
    try:
        payload = verify_token(token)
        if payload:
            return payload.get("sub")
        return None
    except Exception as e:
        logger.error(f"❌ Failed to extract user ID from token: {e}")
        return None


def is_anonymous_token(payload: dict) -> bool:
    """判断是否为匿名用户令牌"""
    return payload.get("type") == "anonymous"


def generate_anonymous_id() -> str:
    """
    生成一个新的匿名用户 ID

    格式: anon_ + 32位随机十六进制字符串
    例如: anon_a1b2c3d4e5f6...

    Returns:
        str: 生成的匿名用户 ID
    """
    random_bytes = uuid.uuid4().bytes + uuid.uuid4().bytes
    hex_str = random_bytes.hex()
    anonymous_id = f"{ANONYMOUS_ID_PREFIX}{hex_str[:ANONYMOUS_ID_LENGTH]}"

    logger.debug(f"✅ Generated new anonymous ID: {anonymous_id[:12]}...")
    return anonymous_id


def is_valid_anonymous_id(anonymous_id: Optional[str]) -> bool:
    """
    验证匿名用户 ID 是否有效（服务端生成的格式）

    有效格式: anon_ + 32位十六进制字符

    Args:
        anonymous_id: 待验证的匿名 ID

    Returns:
        bool: 是否有效
    """
    if not anonymous_id:
        return False

    # 检查前缀
    if not anonymous_id.startswith(ANONYMOUS_ID_PREFIX):
        return False

    # 检查长度
    hex_part = anonymous_id[len(ANONYMOUS_ID_PREFIX):]
    if len(hex_part) != ANONYMOUS_ID_LENGTH:
        return False

    # 检查是否为有效的十六进制
    try:
        int(hex_part, 16)
        return True
    except ValueError:
        return False


def convert_anonymous_id_to_uuid(anonymous_id: str) -> uuid.UUID:
    """
    将服务端生成的匿名 ID 转换为 UUID（用于数据库存储）

    由于匿名 ID 是 32 字节十六进制，我们可以将其转换为 UUID

    Args:
        anonymous_id: 服务端生成的匿名 ID (anon_xxx格式)

    Returns:
        uuid.UUID: 转换后的 UUID
    """
    if not is_valid_anonymous_id(anonymous_id):
        raise ValueError(f"Invalid anonymous ID format: {anonymous_id}")

    hex_part = anonymous_id[len(ANONYMOUS_ID_PREFIX):]

    # 将 32 字节十六进制转换为 UUID（取前 32 字符）
    # UUID 需要 32 个十六进制字符（16 字节）
    uuid_hex = hex_part[:32]
    return uuid.UUID(uuid_hex)