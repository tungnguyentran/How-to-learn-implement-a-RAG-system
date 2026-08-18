# bot/telegram_bot.py
import logging

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from bot.ratelimit import RateLimiter
from config import load_settings
from rag.engine import answer
from rag.llm import LLMError
from storage.db import get_connection
from storage.whitelist import is_whitelisted

logger = logging.getLogger(__name__)

WHITELIST_REJECTION = "Xin lỗi, bạn chưa được cấp quyền sử dụng bot này. Vui lòng liên hệ HR/admin."
RATE_LIMIT_MESSAGE = "Bạn đang gửi câu hỏi quá nhanh, vui lòng chờ một chút rồi thử lại."
GENERIC_ERROR_MESSAGE = "Xin lỗi, hệ thống đang gặp sự cố. Vui lòng thử lại sau."


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Xin chào! Hãy hỏi tôi về các chính sách HR của công ty.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings = context.bot_data["settings"]
    limiter: RateLimiter = context.bot_data["rate_limiter"]
    user_id = update.effective_user.id
    question = update.message.text

    with get_connection(settings) as conn:
        if not is_whitelisted(conn, user_id):
            await update.message.reply_text(WHITELIST_REJECTION)
            return

    if not limiter.allow(user_id):
        await update.message.reply_text(RATE_LIMIT_MESSAGE)
        return

    try:
        with get_connection(settings) as conn:
            result = answer(conn, settings, user_id, question)
    except LLMError:
        logger.exception("LLM call failed for user %s", user_id)
        await update.message.reply_text(GENERIC_ERROR_MESSAGE)
        return

    reply = result.text
    if result.sources:
        sources_block = "\n".join(f"📄 Nguồn: {s}" for s in result.sources)
        reply = f"{reply}\n\n{sources_block}"
    await update.message.reply_text(reply)


async def handle_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Fallback for exceptions unhandled by handle_message (e.g. DB connection
    failures), so the user gets an error reply instead of silence. Not a
    replacement for the deliberate, more-specific LLMError handling above."""
    logger.error("Unhandled error while processing update: %s", update, exc_info=context.error)
    if isinstance(update, Update) and update.message:
        await update.message.reply_text(GENERIC_ERROR_MESSAGE)


def build_application() -> Application:
    settings = load_settings()
    application = Application.builder().token(settings.telegram_bot_token).build()
    application.bot_data["settings"] = settings
    application.bot_data["rate_limiter"] = RateLimiter(settings.rate_limit_per_minute)
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_error_handler(handle_error)
    return application


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    build_application().run_polling()


if __name__ == "__main__":
    main()
