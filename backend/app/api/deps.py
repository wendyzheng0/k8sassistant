
from fastapi import Depends, HTTPException, status, Header, Response, Request
from fastapi.security import OAuth2PasswordBearer
import jwt
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession
import uuid
from typing import Optional

from app.core.config import settings
from app.core.security import (
    generate_anonymous_id,
    is_valid_anonymous_id,
    convert_anonymous_id_to_uuid,
)
from app.core.database import get_session
from app.core.database import get_redis_client
from app.models.models import User
from app.schemas.token import TokenPayload
from app.services.cache_service import CacheService
from app.services.user_service import get_user_service


async def get_cache_service() -> CacheService:
    redis_client = await get_redis_client()
    return CacheService(redis_client)


reusable_oauth2 = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/users/login",
    auto_error=False
)


async def get_current_user(
    token: Optional[str] = Depends(reusable_oauth2),
) -> User:
    user_service = await get_user_service()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        token_data = TokenPayload(**payload)
    except (jwt.PyJWTError, ValidationError):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
        )
    user = await user_service.get_user_by_id(user_id=token_data.sub)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


async def get_current_user_or_anonymous(
    request: Request,
    response: Response,
    anonymous_id: Optional[str] = Header(None, alias="X-Anonymous-User-ID"),  # 保留以兼容旧版前端
    token: Optional[str] = Depends(reusable_oauth2),
) -> uuid.UUID:
    """
    返回认证用户 ID（如果有有效 token）
    否则返回匿名用户 ID（从 Cookie 或生成新的）

    流程:
    1. 首先检查 token，有效则返回认证用户 ID
    2. token 无效时，从 Cookie 读取匿名 ID
    3. Cookie 无或无效时，生成新的匿名 ID 并设置 Cookie
    4. 优先从 Redis 缓存查找用户，缓存未命中才查询数据库
    """
    # 1. 尝试从 token 获取认证用户
    if token:
        try:
            payload = jwt.decode(
                token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
            )
            token_data = TokenPayload(**payload)
            return token_data.sub
        except (jwt.PyJWTError, ValidationError):
            # Token 无效，降级为匿名用户
            pass

    # 2. 处理匿名用户
    user_service = await get_user_service()
    cache_service = None
    try:
        cache_service = await get_cache_service()
    except RuntimeError:
        pass  # Redis 未初始化，继续使用数据库

    # 从 Cookie 获取匿名 ID
    anon_id_from_cookie = request.cookies.get("anonymous_id")

    # 使用优先级：Cookie > Header > 新生成
    final_anon_id = None
    is_new_anon_id = False

    if anon_id_from_cookie and is_valid_anonymous_id(anon_id_from_cookie):
        final_anon_id = anon_id_from_cookie
    elif anonymous_id and is_valid_anonymous_id(anonymous_id):
        # Header 中的 ID 也需要验证格式（防止伪造）
        final_anon_id = anonymous_id
    else:
        # 生成新的匿名 ID
        final_anon_id = generate_anonymous_id()
        is_new_anon_id = True

    # 3. 将匿名 ID 转换为 UUID
    try:
        anonymous_uuid = convert_anonymous_id_to_uuid(final_anon_id)
    except ValueError:
        # 转换失败，生成新的
        final_anon_id = generate_anonymous_id()
        anonymous_uuid = convert_anonymous_id_to_uuid(final_anon_id)
        is_new_anon_id = True

    # 4. 优先从 Redis 缓存查找用户
    user = None
    cached_user_id = None

    if cache_service:
        # 从 Redis 获取匿名 ID -> 用户 ID 的映射
        cached_user_id = await cache_service.get_user_by_anonymous_id(anonymous_uuid)
        if cached_user_id:
            # 从 Redis 获取用户信息
            cached_user_data = await cache_service.get_cached_user(uuid.UUID(cached_user_id))
            if cached_user_data:
                # 从缓存数据构建用户对象
                from datetime import datetime
                user = User(
                    id=uuid.UUID(cached_user_data['id']),
                    username=cached_user_data['username'],
                    email=cached_user_data['email'],
                    is_active=cached_user_data['is_active'],
                    is_anonymous=cached_user_data['is_anonymous'],
                    anonymous_id=uuid.UUID(cached_user_data['anonymous_id']) if cached_user_data.get('anonymous_id') else None,
                    created_at=datetime.fromisoformat(cached_user_data['created_at']) if cached_user_data.get('created_at') else None,
                    updated_at=datetime.fromisoformat(cached_user_data['updated_at']) if cached_user_data.get('updated_at') else None,
                )

    # 5. 缓存未命中，查询数据库
    if not user:
        user = await user_service.get_user_by_anonymous_id(anonymous_uuid)

        if not user:
            # 创建新的匿名用户（会自动缓存）
            user = await user_service.create_anonymous_user(anonymous_id=anonymous_uuid)
            is_new_anon_id = True

    # 6. 设置 Cookie（如果是新生成的或需要更新）
    if is_new_anon_id or anon_id_from_cookie != final_anon_id:
        response.set_cookie(
            key="anonymous_id",
            value=final_anon_id,
            max_age=30 * 24 * 60 * 60,  # 30 天
            expires=30 * 24 * 60 * 60,   # 30 天
            path="/",
            httponly=True,               # 防止 XSS 访问
            samesite="lax",              # CSRF 保护
            secure=False,                # 开发环境用 False，生产环境用 True
        )

    return user.id
