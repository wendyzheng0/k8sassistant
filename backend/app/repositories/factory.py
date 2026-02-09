from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.user_repository import UserRepository
from app.repositories.chat_repository import ChatRepository
from app.repositories.message_repository import MessageRepository


class RepositoryFactory:
    def __init__(self, session: AsyncSession):
        self.session = session
        self._user_repository = None
        self._chat_repository = None
        self._message_repository = None

    @property
    def users(self) -> UserRepository:
        """用户仓库属性"""
        if self._user_repository is None:
            self._user_repository = UserRepository(self.session)
        return self._user_repository

    @property
    def chats(self) -> ChatRepository:
        """聊天仓库属性"""
        if self._chat_repository is None:
            self._chat_repository = ChatRepository(self.session)
        return self._chat_repository

    @property
    def messages(self) -> MessageRepository:
        """消息仓库属性"""
        if self._message_repository is None:
            self._message_repository = MessageRepository(self.session)
        return self._message_repository

    def get_user_repository(self) -> UserRepository:
        return UserRepository(self.session)

    def get_chat_repository(self) -> ChatRepository:
        return ChatRepository(self.session)

    def get_message_repository(self) -> MessageRepository:
        return MessageRepository(self.session)

    async def commit(self) -> None:
        """提交事务"""
        await self.session.commit()

    async def rollback(self) -> None:
        """回滚事务"""
        await self.session.rollback()

    async def close(self) -> None:
        """关闭会话"""
        await self.session.close()
