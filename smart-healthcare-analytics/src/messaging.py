from __future__ import annotations

from datetime import datetime
from typing import List, Dict

import pandas as pd
from sqlalchemy import or_, select, and_

from .database import session_scope, User, Message


def send_message(sender_id: int, receiver_id: int, content: str) -> None:
    """Persist a message between two users."""
    text = (content or "").strip()
    if not text or not sender_id or not receiver_id:
        return

    with session_scope() as session:
        sender = session.get(User, sender_id)
        receiver = session.get(User, receiver_id)
        if not sender or not receiver:
            return

        msg = Message(
            sender_id=sender.id,
            receiver_id=receiver.id,
            content=text,
            timestamp=datetime.utcnow(),
        )
        session.add(msg)


def get_conversation(user_a_id: int, user_b_id: int, limit: int = 200) -> pd.DataFrame:
    """Return ordered messages between two users."""
    if not user_a_id or not user_b_id:
        return pd.DataFrame()

    with session_scope() as session:
        stmt = (
            select(Message).where(
                or_(
                    and_(Message.sender_id == user_a_id, Message.receiver_id == user_b_id),
                    and_(Message.sender_id == user_b_id, Message.receiver_id == user_a_id),
                )
            )
            .order_by(Message.timestamp.asc())
        )
        messages: List[Message] = session.execute(stmt).scalars().all()
        if limit and len(messages) > limit:
            messages = messages[-limit:]

        rows: List[Dict] = []
        for m in messages:
            rows.append(
                {
                    "id": m.id,
                    "sender_id": m.sender_id,
                    "receiver_id": m.receiver_id,
                    "content": m.content,
                    "timestamp": m.timestamp,
                }
            )
    return pd.DataFrame(rows)

