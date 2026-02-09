
import uuid
from fastapi import APIRouter, Depends, HTTPException, status

from app.services.user_service import UserService, get_user_service
from app.schemas.user import UserCreate, UserLogin, User, UserMerge
from app.schemas.token import Token
from app.core.security import create_access_token
from app.api.deps import get_current_user

router = APIRouter()


@router.post("/register", response_model=User)
async def register_user(user_in: UserCreate):
    user_service = await get_user_service()
    user = await user_service.get_user_by_email(email=user_in.email)
    if user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The user with this email already exists in the system.",
        )
    user = await user_service.create_user(user_in)
    return user


@router.post("/login", response_model=Token)
async def login_for_access_token(form_data: UserLogin):
    user_service = await get_user_service()
    user = await user_service.authenticate_user(
        email=form_data.username, password=form_data.password
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": str(user.id)})
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/me", response_model=User)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    return current_user


@router.get("/{user_id}", response_model=User)
async def get_user_by_id(
    user_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
):
    user_service = await get_user_service()
    user = await user_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    return user


@router.put("/{user_id}", response_model=User)
async def update_user_profile(
    user_id: uuid.UUID,
    user_update: dict,
    current_user: User = Depends(get_current_user),
):
    # Verify ownership - users can only update their own profile
    if current_user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only update your own profile",
        )
    user_service = await get_user_service()
    success = await user_service.update_user_profile(user_id, **user_update)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to update user profile",
        )
    # Return updated user
    updated_user = await user_service.get_user_by_id(user_id)
    return updated_user


@router.delete("/{user_id}")
async def delete_user_account(
    user_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
):
    # Verify ownership - users can only delete their own account
    if current_user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only delete your own account",
        )
    user_service = await get_user_service()
    success = await user_service.delete_user(user_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to delete user account",
        )
    return {"message": "User account deleted successfully"}


@router.post("/merge")
async def merge_user_data(
    merge_in: UserMerge,
    current_user: User = Depends(get_current_user),
):
    user_service = await get_user_service()
    success = await user_service.merge_anonymous_user(
        anonymous_user_id=merge_in.anonymous_user_id, login_user_id=current_user.id
    )
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to merge user data.",
        )
    return {"message": "User data merged successfully."}


@router.post("/logout")
async def logout_user(
    current_user: User = Depends(get_current_user),
):
    user_service = await get_user_service()
    success = await user_service.logout(current_user.id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to logout user.",
        )
    return {"message": "User logged out successfully."}
